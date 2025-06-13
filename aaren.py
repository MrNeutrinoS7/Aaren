import torch
import torch.nn as nn

class Aaren(nn.Module):
    def __init__(self, d_model):
        self.d_model = d_model
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
    
    def concatenate(self, tuple_i, tuple_j):
        """Numerically stable merge of proposed tuples"""
        m_i, u_i, w_i = tuple_i
        m_j, u_j, w_j = tuple_j
        
        m_f = torch.maximum(m_i, m_j)
        exp_diff_i = torch.exp(m_i - m_f)
        exp_diff_j = torch.exp(m_j - m_f)
        
        u_f = u_i * exp_diff_i + u_j * exp_diff_j
        w_f = w_i * exp_diff_i + w_j * exp_diff_j
        
        return (m_f, u_f, w_f)
    
    def forward(self, K, V, mode=0):
        s = torch.matmul(K, self.q)
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
            return torch.zeros(self.d_model,self.d_model)
            #return self.parallel_forward(s, V)
            #I don't know how to implement parralel scan correctly
            #I tried but it had even worse performance than the iterative implementation, bruh
        else:
            raise ValueError("Invalid mode")
            

class AtteRNNtion(nn.Module):
    def __init__(self, d_model, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([Aaren(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, X, mode=0):
        K = self.W_k(X)
        V = self.W_v(X)
        for layer in self.layers:
            H = layer(K, V, mode)
            K = self.W_k(H)
            V = self.W_v(H)  
        return self.norm(H)

    def fit(self, dataloader, epochs=10, lr=1e-4, clip_grad=1.0):
        """Training loop that handles device placement and gradient clipping."""
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch in dataloader:
                # Auto-detect batch type (single tensor or tuple)
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    x, y = batch.to(device), None  # Assume autoencoder
                
                optimizer.zero_grad()
                logits = self(x, mode=1)  # Parallel mode for training
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)) if y else logits.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
            

