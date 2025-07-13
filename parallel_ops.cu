#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ######################## UTILITY MACROS ########################
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err));                         \
        exit(1);                                                \
    }                                                           \
} while(0)

// ######################## FORWARD KERNEL ########################
template <typename scalar_t>
__global__ void parallel_forward_kernel(
    const scalar_t* __restrict__ s,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ out_float,
    scalar_t* __restrict__ out_int,
    scalar_t* __restrict__ out_vec,
    int N, int D) {
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // Initialize outputs
    out_float[j] = s[j];
    out_int[j] = scalar_t(1);
    
    scalar_t* vec_out = out_vec + j * D;
    const scalar_t* vec_in = V + j * D;
    
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        vec_out[d] = vec_in[d];
    }

    __syncthreads();

    // Parallel reduction with log2 steps
    for (int i = 0; i <= (int)log2f((float)N); ++i) {
        const int stride = 1 << i;
        if (j >= stride && (j - stride) >= 0) {
            const int src = j - stride;
            
            const scalar_t m_i = out_float[j];
            const scalar_t m_j = out_float[src];
            const scalar_t m_f = max(m_i, m_j);
            
            const scalar_t exp_i = exp2f(m_i - m_f);
            const scalar_t exp_j = exp2f(m_j - m_f);
            
            // Update outputs
            out_float[j] = m_f;
            out_int[j] = out_int[j] * exp_i + out_int[src] * exp_j;
            
            const scalar_t* src_vec = out_vec + src * D;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                vec_out[d] = vec_out[d] * exp_i + src_vec[d] * exp_j;
            }
        }
        __syncthreads();
    }
}

// ######################## BACKWARD KERNELS ########################
template <typename scalar_t>
__global__ void backward_input_kernel(
    const scalar_t* __restrict__ s,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ out_vec,
    const scalar_t* __restrict__ out_int,
    scalar_t* __restrict__ grad_in,
    int N, int D) {
    
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(smem);
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // 1. Find row maximum (first half of shared mem)
    scalar_t max_val = -INFINITY;
    for (int i = 0; i <= j; ++i) {
        max_val = max(max_val, s[i]);
    }
    shared[threadIdx.x] = max_val;
    __syncthreads();

    // Parallel reduction for max
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const scalar_t row_max = shared[0];
    __syncthreads();

    // 2. Compute sum exps (second half of shared mem)
    scalar_t sum_exp = 0;
    for (int i = 0; i <= j; ++i) {
        sum_exp += exp2f(s[i] - row_max);
    }
    shared[threadIdx.x] = sum_exp;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    const scalar_t norm = 1.0f / shared[0];
    __syncthreads();

    // 3. Compute gradient
    scalar_t grad = 0;
    int argmax = 0;
    scalar_t max_s = -INFINITY;
    
    const scalar_t* g_out = grad_out + j * D;
    
    for (int i = 0; i <= j; ++i) {
        const scalar_t* v = V + i * D;
        scalar_t dot = 0;
        
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += g_out[d] * v[d];
        }
        
        const scalar_t exp_term = exp2f(s[i] - row_max) * norm;
        grad += dot * exp_term;
        
        if (s[i] > max_s) {
            max_s = s[i];
            argmax = i;
        }
    }
    
    // Subtract argmax term
    scalar_t argmax_dot = 0;
    const scalar_t* argmax_v = V + argmax * D;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        argmax_dot += g_out[d] * argmax_v[d];
    }
    
    grad_in[j] = grad - argmax_dot * out_int[j];
}

template <typename scalar_t>
__global__ void backward_weight_kernel(
    const scalar_t* __restrict__ s,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_weight,
    int N, int D) {
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    for (int i = j; i < N; ++i) {
        // Find row max
        scalar_t max_val = -INFINITY;
        for (int k = 0; k <= i; ++k) {
            max_val = max(max_val, s[k]);
        }
        
        // Compute sum exps
        scalar_t sum_exp = 0;
        for (int k = 0; k <= i; ++k) {
            sum_exp += exp2f(s[k] - max_val);
        }
        
        const scalar_t norm = 1.0f / sum_exp;
        const scalar_t* g_out = grad_out + i * D;
        
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            atomicAdd(&grad_weight[j * D + d], 
                     g_out[d] * exp2f(s[j] - max_val) * norm);
        }
    }
}

// ######################## HOST FUNCTIONS ########################
std::vector<torch::Tensor> parallel_forward(
    torch::Tensor s,
    torch::Tensor V) {
    
    CHECK_INPUT(s);
    CHECK_INPUT(V);
    TORCH_CHECK(s.dim() == 1, "s must be 1D");
    TORCH_CHECK(V.dim() == 2, "V must be 2D");
    TORCH_CHECK(s.size(0) == V.size(0), "Dimension mismatch");

    const int N = s.size(0);
    const int D = V.size(1);
    
    auto out_float = torch::empty_like(s);
    auto out_int = torch::empty_like(s);
    auto out_vec = torch::empty_like(V);

    const int threads = std::min(512, N);
    const int blocks = (N + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(s.scalar_type(), "forward", ([&] {
        parallel_forward_kernel<scalar_t><<<blocks, threads>>>(
            s.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            out_float.data_ptr<scalar_t>(),
            out_int.data_ptr<scalar_t>(),
            out_vec.data_ptr<scalar_t>(),
            N, D);
    }));
    
    CUDA_CHECK(cudaGetLastError());
    return {out_float, out_int, out_vec};
}

std::vector<torch::Tensor> parallel_backward(
    torch::Tensor s,
    torch::Tensor V,
    torch::Tensor grad_out,
    torch::Tensor out_vec,
    torch::Tensor out_int) {
    
    CHECK_INPUT(s); CHECK_INPUT(V); 
    CHECK_INPUT(grad_out); CHECK_INPUT(out_vec); CHECK_INPUT(out_int);
    
    const int N = s.size(0);
    const int D = V.size(1);
    
    auto grad_in = torch::zeros_like(s);
    auto grad_weight = torch::zeros_like(V);

    // Launch input gradient kernel
    const int threads1 = std::min(256, N);
    const int blocks1 = (N + threads1 - 1) / threads1;
    const size_t smem_size = 2 * threads1 * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(s.scalar_type(), "backward_input", ([&] {
        backward_input_kernel<scalar_t><<<blocks1, threads1, smem_size>>>(
            s.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            out_vec.data_ptr<scalar_t>(),
            out_int.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            N, D);
    }));

    // Launch weight gradient kernel
    const int threads2 = std::min(256, N);
    const int blocks2 = (N + threads2 - 1) / threads2;
    
    AT_DISPATCH_FLOATING_TYPES(s.scalar_type(), "backward_weight", ([&] {
        backward_weight_kernel<scalar_t><<<blocks2, threads2>>>(
            s.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            N, D);
    }));

    CUDA_CHECK(cudaGetLastError());
    return {grad_in, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &parallel_forward, "Parallel forward pass");
    m.def("backward", &parallel_backward, "Parallel backward pass");
}
