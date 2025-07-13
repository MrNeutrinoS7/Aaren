import torch
import torch.nn as nn
import parallel_ops

class ParallelForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, V):
        # Ensure all tensors are on the same device
        device = s.device
        V = V.to(device)
        
        output_scalar_float, output_scalar_int, output_vector = parallel_ops.forward(s, V)
        ctx.save_for_backward(s, V, output_vector, output_scalar_int)
        return output_vector
    
    @staticmethod
    def backward(ctx, grad_output):
        s, V, output_vector, output_scalar_int = ctx.saved_tensors
        grad_input, grad_weight = parallel_ops.backward(s, V, grad_output.to(s.device), 
                                 output_vector, output_scalar_int)
        return grad_input, grad_weight

def parallel_forward(s, V):
    return ParallelForward.apply(s, V)

class Aaren(nn.Module):
    def __init__(self, d_model, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.q = nn.Parameter(torch.randn(d_model, device=device))*10
        self.memory = (
            torch.tensor(float('-inf'), device=self.device),
            torch.tensor(0.0, device=self.device), 
            torch.zeros(self.d_model, device=self.device)
        )
    def concatenate(self, tuple_i, tuple_j):
        m_i, u_i, w_i = tuple_i
        m_j, u_j, w_j = tuple_j
        device = m_i.device
        m_j = m_j.to(device)
        u_j = u_j.to(device)
        w_j = w_j.to(device)
        m_f = torch.maximum(m_i, m_j)
        exp_diff_i = torch.exp(m_i - m_f)
        exp_diff_j = torch.exp(m_j - m_f)
        u_f = u_i * exp_diff_i + u_j * exp_diff_j
        w_f = w_i * exp_diff_i + w_j * exp_diff_j
        return (m_f, u_f, w_f)

    def compute(self, next):
        device = self.memory[0].device
        next = (next[0].to(device), next[1].to(device), next[2].to(device))
        return self.concatenate(self.memory, next)

    def forward(self, K, V, mode="iterative"):
        device = self.q.device
        K = K.to(device)
        V = V.to(device)
        output = torch.empty_like(V, device=device)
        s = torch.stack([torch.dot(k, self.q) for k in K])
        
        if mode == "iterative":
            output_list = []
            for i in range(len(V)):
                next_tuple = (s[i], torch.tensor(1.0, device=device), V[i])
                self.memory = self.compute(next_tuple)
                output_list.append(self.memory[2] / self.memory[1])
            output = torch.stack(output_list)
                
        elif mode == "parallel":
            final_vectors = parallel_forward(s, V)
            last = (s[-1], torch.tensor(1.0, device=device), V[-1])
            self.memory = self.concatenate(self.memory, last)
            output = final_vectors
                
        return output                      
                                          
class AtteRNNtion(nn.Module):
    def __init__(self, d_model, default_mode="iterative", num_layers=1):
        super().__init__()
        self.mode = default_mode
        self.d_model = d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([Aaren(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, X):
        device = next(self.parameters()).device
        
        K = self.W_k(X.to(device))
        V = self.W_v(X.to(device))
        
        for layer in self.layers:
            layer.to(device)
            H = layer(K, V, self.mode)
            K = self.W_k(H)
            V = self.W_v(H)
            
        return self.norm(H)
 


