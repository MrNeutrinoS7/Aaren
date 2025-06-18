import torch
import torch.nn as nn
import math
import numpy as np
from numba import cuda

    
@cuda.jit
def update_kernel(i : int, N : int, temporary : np.array, output : np.array):
    j = cuda.grid(1)
    if j < N-1:
        if j < math.exp(i):
            temporary[j] = output[j]
        else:
            m_i = output[j][0]
            u_i = output[j][1]
            w_i = output[j][2]
            
            m_j = output[j-int(math.exp(i))][0]
            u_j = output[j-int(math.exp(i))][1]
            w_j = output[j-int(math.exp(i))][2]
            
            m_f = m_i if m_i > m_j else m_j
            exp_diff_i = math.exp(m_i - m_f)
            exp_diff_j = math.exp(m_j - m_f)
            
            u_f = u_i * exp_diff_i + u_j * exp_diff_j
            # Set scalar fields
            temporary[j][0] = m_f
            temporary[j][1] = u_f
            
            # Set vector elements individually
            for d in range(len(w_i)):
                temporary[j][2][d] = w_i[d] * exp_diff_i + w_j[d] * exp_diff_j
            
            
def parallel_forward(s : np.array, V : np.array) -> np.array:
    ttype = np.dtype([
    ('scalar_float', np.float32),
    ('scalar_int', np.int32),
    ('vector', np.float32, (len(V[0]),))  
    ])
    N = len(s)
    neutral = (np.float32(0.0), np.int32(0), np.zeros(len(V[0]), dtype=np.float32))
    temporary = np.array([neutral for _ in range(N)], dtype=ttype)
    output = np.array([(s_i,1,v_i) for s_i,v_i in zip(s, V)],dtype=ttype)
    for i in range(0,int(math.log(N))):
        threads_per_block = 256
        blocks_per_grid = int((N + threads_per_block - 1) // threads_per_block)
        update_kernel[blocks_per_grid,threads_per_block](i, N, temporary, output)
        output = temporary
    return output


class Aaren(nn.Module):
    def __init__(self, d_model):
        self.d_model = d_model
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
    
    def concatenate(self, tuple_i, tuple_j):
        """Numerically stable merge of proposed tuples"""
        m_i, u_i, w_i = tuple_i
        m_j, u_j, w_j = tuple_j
        
        m_f = max(m_i, m_j)
        exp_diff_i = math.exp(m_i - m_f)
        exp_diff_j = math.exp(m_j - m_f)
        
        u_f = u_i * exp_diff_i + u_j * exp_diff_j
        w_f = w_i * exp_diff_i + w_j * exp_diff_j
        
        return (m_f, u_f, w_f)
    
        
    def forward(self, K : torch.tensor, V : torch.tensor, mode=0):
        s = [torch.dot(k,self.q) for k in K]
        s = torch.tensor(s)
        if mode == 0:
            output = torch.empty_like(V)
            m_prev = torch.tensor(float('-inf'), device=s.device)
            u_prev = torch.tensor(0.0, device=s.device)
            w_prev = torch.zeros_like(V[0])
            
            for i in range(len(K)):
                m_curr = torch.maximum(m_prev, s[i])
                exp_prev = torch.exp(m_prev - m_curr)
                exp_curr = torch.exp(s[i] - m_curr)
                
                u_curr = u_prev * exp_prev + exp_curr
                w_curr = w_prev * exp_prev + V[i] * exp_curr
                
                output[i] = w_curr / u_curr
                m_prev, u_prev, w_prev = m_curr, u_curr, w_curr
            return output
        elif mode == 1:
            s = s.numpy()
            V = V.numpy()
            return torch.tensor(parallel_forward(s, V))
        else:
            raise ValueError("Invalid mode")

            

