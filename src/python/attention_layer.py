import torch
import torch.nn as nn
import numpy as np
import sys
import time

# Module availability flags
ATTENTION_MODULE_AVAILABLE = False
CUDA_AVAILABLE = False
OPENMP_AVAILABLE = False
attention_cuda_py = None

def try_import_attention_module():
    """Try to import the attention module with proper error handling"""
    global ATTENTION_MODULE_AVAILABLE, CUDA_AVAILABLE, OPENMP_AVAILABLE, attention_cuda_py
    
    try:
        import attention_cuda_py
        ATTENTION_MODULE_AVAILABLE = True
        
        try:
            CUDA_AVAILABLE = attention_cuda_py.cuda_available()
            OPENMP_AVAILABLE = attention_cuda_py.openmp_available()
            
            print(f"✓ Attention module loaded successfully!")
            print(f"  - CUDA available: {CUDA_AVAILABLE}")
            print(f"  - OpenMP available: {OPENMP_AVAILABLE}")
            
            if CUDA_AVAILABLE:
                print(f"  - CUDA devices: {attention_cuda_py.get_cuda_device_count()}")
            
            if OPENMP_AVAILABLE:
                print(f"  - OpenMP max threads: {attention_cuda_py.get_openmp_max_threads()}")
                
        except Exception as e:
            print(f"Warning: Module availability check failed: {e}")
            CUDA_AVAILABLE = False
            OPENMP_AVAILABLE = False
            
    except ImportError as e:
        print(f"Info: Custom attention module not available ({e})")
        print("Falling back to PyTorch attention")
        ATTENTION_MODULE_AVAILABLE = False
        CUDA_AVAILABLE = False
        OPENMP_AVAILABLE = False
        attention_cuda_py = None
    except Exception as e:
        print(f"Error: Unexpected error loading attention module: {e}")
        ATTENTION_MODULE_AVAILABLE = False
        CUDA_AVAILABLE = False
        OPENMP_AVAILABLE = False
        attention_cuda_py = None

# Try to import the module
try_import_attention_module()

class AttentionBackend:
    """Helper class to manage attention backend selection"""
    
    @staticmethod
    def compute_attention(Q, K, V, backend='auto', num_heads=1, causal_mask=False):
        """
        Compute attention using the specified backend
        
        Args:
            Q, K, V: numpy arrays of shape [seq_len, embed_dim]
            backend: str, one of 'auto', 'cuda', 'openmp', 'cpu', 'pytorch'
            num_heads: int, number of attention heads (for multi-head attention)
            causal_mask: bool, whether to apply causal masking
            
        Returns:
            numpy array of shape [seq_len, embed_dim] or None if failed
        """
        if not ATTENTION_MODULE_AVAILABLE:
            return None
        
        # Auto-select best available backend
        if backend == 'auto':
            if CUDA_AVAILABLE:
                backend = 'cuda'
            elif OPENMP_AVAILABLE:
                backend = 'openmp'
            else:
                backend = 'cpu'
        
        try:
            if backend == 'cuda' and CUDA_AVAILABLE:

                result = attention_cuda_py.attention_cuda(Q, K, V)
                return result
                    
            elif backend == 'openmp' and OPENMP_AVAILABLE:
                result = attention_cuda_py.attention_openmp(Q, K, V)
                return result
                
            elif backend == 'cpu':
                result = attention_cuda_py.attention_cpu(Q, K, V)
                return result
                
            else:
                print(f"Backend '{backend}' not available or not supported")
                return None
                
        except Exception as e:
            print(f"Backend '{backend}' failed: {e}")
            return None

class EnhancedAttentionLayer(nn.Module):
    """Enhanced attention layer with CUDA, OpenMP, and CPU implementations"""
    
    def __init__(self, embed_dim, num_heads=1, use_cuda=True, use_openmp=True, 
                 max_seq_len=4096, backend='auto', causal_mask=False):
        super(EnhancedAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.use_openmp = use_openmp and OPENMP_AVAILABLE
        self.max_seq_len_for_custom = max_seq_len
        self.backend = backend
        self.causal_mask = causal_mask
        
        # Validate num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        
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
        
            
        # Process batch using custom attention
        outputs = []
        backend_used = None
        
        for i in range(batch_size):
            q_i = Q[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
            k_i = K[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
            v_i = V[i].detach().cpu().numpy().astype(np.float32)  # [L, D]
            
            # Try custom attention backends
            result = AttentionBackend.compute_attention(
                q_i, k_i, v_i, 
                backend=self.backend, 
                num_heads=self.num_heads,
                causal_mask=self.causal_mask
            )
            
            result_tensor = torch.from_numpy(result).to(x.device)
            outputs.append(result_tensor)
            if backend_used is None:
                # Determine which backend was actually used
                if self.backend == 'auto':
                    if CUDA_AVAILABLE :
                        backend_used = 'cuda'
                    elif OPENMP_AVAILABLE:
                        backend_used = 'openmp'
                    else:
                        backend_used = 'cpu'
                else:
                    backend_used = self.backend
            
        
        output = torch.stack(outputs)
        return self.out_proj(output)
    
    
class SpatialAttentionLayer(nn.Module):
    """2D spatial attention for CNN feature maps with backend selection"""
    
    def __init__(self, in_channels, use_cuda=True, use_openmp=True, 
                 max_spatial_size=4096, backend='auto'):
        super(SpatialAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.use_openmp = use_openmp and OPENMP_AVAILABLE
        self.max_spatial_size = max_spatial_size
        self.backend = backend
        
        # 1x1 convolutions for Q, K, V
        self.q_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        spatial_size = height * width
        
        # Generate Q, K, V
        Q = self.q_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        K = self.k_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        V = self.v_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        
        
        outputs = []
        for i in range(batch_size):
            q_i = Q[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
            k_i = K[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
            v_i = V[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
            
            # Try custom attention backends
            result = AttentionBackend.compute_attention(
                q_i, k_i, v_i, backend=self.backend
            )
            
            result_tensor = torch.from_numpy(result).to(x.device)
            outputs.append(result_tensor)
                    
        output = torch.stack(outputs)
        
        # Reshape back to spatial dimensions
        output = output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return self.out_conv(output)
    


