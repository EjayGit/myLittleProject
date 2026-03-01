import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum # Einstein summation -> matrix maths using strings
import math

class SimplifiedMamba(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, d_state=16, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create residual blocks
        self.layers = nn.ModuleList([
            MambaResidualBlock(d_model, d_state, expand) 
            for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Tie weights
        self.head.weight = self.embedding.weight
        
        # Enhanced initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.embedding.num_embeddings,
            'hidden_dim': self.embedding.embedding_dim,
            'num_layers': len(self.layers)
        }

class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state, expand):
        super().__init__()
        self.mixer = MambaBlock(d_model, d_state, expand)
        self.norm = RMSNorm(d_model)
        
    def forward(self, x):
        return x + self.mixer(self.norm(x))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, expand):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # SSM parameters
        self._init_A_stable(d_state=d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        
        # Initialize dt_proj bias to 1
        nn.init.constant_(self.dt_proj.bias, 1.0)
        
        # Enhanced weight initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize in_proj
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        
        # Initialize out_proj  
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.d_inner))
        nn.init.zeros_(self.out_proj.bias)
        
        # Initialize x_proj
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.x_proj.bias)
        
        # Initialize dt_proj
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.dt_proj.bias, 1.0)

    def _init_A_stable(self, d_state):
        # Create a simple, stable diagonal matrix
        A = torch.zeros(self.d_inner, d_state)
        
        # Fill diagonal with stable negative values
        for i in range(min(self.d_inner, d_state)):
            # Stable geometric decay - all values between -1.5 and -0.5
            A[i, i] = -1.0 - (i * 0.01)
        
        # Ensure all values are negative and reasonable
        A = torch.clamp(A, min=-2.0, max=-0.1)
        
        # Convert to log space safely
        self.A_log = nn.Parameter(torch.log(-A))
        
        print(f"STABLE A_log initialized: min={self.A_log.min().item():.4f}, max={self.A_log.max().item():.4f}")
        print(f"A matrix will be: min={A.min().item():.4f}, max={A.max().item():.4f}")

    def ssm(self, x):
        # SSM with NaN protection
        b, l, d = x.shape
        n = self.d_state
        
        # Get selective parameters with stability
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, n, n], dim=-1)
        
        # Stable delta processing
        delta = F.softplus(self.dt_proj(delta)) + 1e-8
        
        # Discretization with NaN protection
        A = -torch.exp(self.A_log.float())  # Use float for stability
        
        # Check for NaN and fix if needed
        if torch.isnan(A).any():
            print("NaN detected in A, using fallback initialization")
            A = -torch.ones_like(A) * 0.9
        
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        
        # Sequential scan with stability
        state = torch.zeros(b, d, n, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(l):
            # Ultra-stable state update with clamping
            deltaA_slice = torch.clamp(deltaA[:, i], min=1e-12, max=1e4)
            deltaB_u_slice = torch.clamp(deltaB_u[:, i], min=-1e4, max=1e4)
            
            state = deltaA_slice * state + deltaB_u_slice
            
            # Prevent state explosion
            state = torch.clamp(state, min=-1e6, max=1e6)
            
            y_i = einsum(state, C[:, i, :], 'b d n, b n -> b d')
            y_i = torch.clamp(y_i, min=-1e6, max=1e6)
            
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)
        
        # Stable skip connection
        D_safe = torch.clamp(self.D.unsqueeze(0).unsqueeze(0), min=-1e2, max=1e2)
        output = y + x * D_safe
        
        return output

    def forward(self, x):
        b, l, _ = x.shape
        
        # Project input and split into two branches
        x_proj = self.in_proj(x)
        x, res = x_proj.split(self.d_inner, dim=-1)
        
        # Apply activation to first branch
        x = F.silu(x)
        
        # Process through SSM
        y = self.ssm(x)
        
        # Gated output with second branch
        y = y * F.silu(res)
        
        # Final projection
        output = self.out_proj(y)
        
        # Check for NaN
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Numerical instability in MambaBlock, returning zeros")
            output = torch.zeros_like(output)
        
        return output
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): # 
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
