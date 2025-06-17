import torch
import torch.nn as nn
import numpy as np
import sys

# Try to import CUDA module
try:
    import attention_cuda_py
    CUDA_AVAILABLE = attention_cuda_py.cuda_available()
    print(f"CUDA attention module available: {CUDA_AVAILABLE}")
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA attention module not available, falling back to PyTorch")

class CUDAAttentionLayer(nn.Module):
    """Custom attention layer with both CPU and CUDA implementations"""
    
    def __init__(self, embed_dim, num_heads=1, use_cuda=True):
        super(CUDAAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        print(f"Attention layer initialized with CUDA: {self.use_cuda}")
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.q_proj(x)  # [B, L, D]
        K = self.k_proj(x)  # [B, L, D]
        V = self.v_proj(x)  # [B, L, D]
        
        if self.use_cuda and seq_len <= 4096:  # Only use CUDA for reasonable sequence lengths
            # Process each batch item separately
            outputs = []
            for i in range(batch_size):
                q_i = Q[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
                k_i = K[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
                v_i = V[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
                
                try:
                    result = attention_cuda_py.attention_cuda(q_i, k_i, v_i)
                    result_tensor = torch.from_numpy(result).to(x.device)
                    outputs.append(result_tensor)
                except Exception as e:
                    print(f"CUDA attention failed: {e}, falling back to PyTorch")
                    result_tensor = self._pytorch_attention(Q[i:i+1], K[i:i+1], V[i:i+1]).squeeze(0)
                    outputs.append(result_tensor)
            
            output = torch.stack(outputs)
        else:
            output = self._pytorch_attention(Q, K, V)
        
        return self.out_proj(output)
    
    def _pytorch_attention(self, Q, K, V):
        """PyTorch attention implementation"""
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, L, L]
        scores = scores / np.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, L, L]
        output = torch.matmul(attn_weights, V)  # [B, L, D]
        return output

class SpatialAttentionLayer(nn.Module):
    """2D spatial attention for CNN feature maps"""
    
    def __init__(self, in_channels, use_cuda=True):
        super(SpatialAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        print(f"Spatial attention layer initialized with CUDA: {self.use_cuda}")
        
        # 1x1 convolutions for Q, K, V
        self.q_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        
        # Generate Q, K, V
        Q = self.q_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        K = self.k_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        V = self.v_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        
        if self.use_cuda and seq_len <= 4096:  # Only use CUDA for reasonable sequence lengths
            outputs = []
            for i in range(batch_size):
                q_i = Q[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                k_i = K[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                v_i = V[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                
                try:
                    result = attention_cuda_py.attention_cuda(q_i, k_i, v_i)
                    result_tensor = torch.from_numpy(result).to(x.device)
                    outputs.append(result_tensor)
                except Exception as e:
                    print(f"CUDA attention failed: {e}, falling back to PyTorch")
                    result_tensor = self._pytorch_attention(Q[i:i+1], K[i:i+1], V[i:i+1]).squeeze(0)
                    outputs.append(result_tensor)
            
            output = torch.stack(outputs)
        else:
            output = self._pytorch_attention(Q, K, V)
        
        # Reshape back to spatial dimensions
        output = output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return self.out_conv(output)
    
    def _pytorch_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, HW, HW]
        scores = scores / np.sqrt(self.in_channels)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, HW, HW]
        output = torch.matmul(attn_weights, V)  # [B, HW, C]
        return output