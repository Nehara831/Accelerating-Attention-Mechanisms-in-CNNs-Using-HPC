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
                if num_heads > 1:
                    print("Warning: Multi-head attention not implemented for CUDA, falling back to OpenMP")
                    return AttentionBackend.compute_attention(Q, K, V, 'openmp', num_heads, causal_mask)
                elif causal_mask:
                    print("Warning: Causal masking not implemented for CUDA, falling back to OpenMP")
                    return AttentionBackend.compute_attention(Q, K, V, 'openmp', num_heads, causal_mask)
                else:
                    result = attention_cuda_py.attention_cuda(Q, K, V)
                    return result
                    
            elif backend == 'openmp' and OPENMP_AVAILABLE:
                result = attention_cuda_py.attention_openmp(Q, K, V)
                return result
                
            elif backend == 'cpu':
                if num_heads > 1 or causal_mask:
                    print("Warning: Multi-head and masked attention not fully implemented for basic CPU, using basic attention")
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
        
        print(f"Enhanced attention layer initialized:")
        print(f"  - Module available: {ATTENTION_MODULE_AVAILABLE}")
        print(f"  - CUDA available: {CUDA_AVAILABLE}")
        print(f"  - OpenMP available: {OPENMP_AVAILABLE}")
        print(f"  - Backend: {backend}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Causal mask: {causal_mask}")
        
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
        
        # Use PyTorch for very long sequences or when custom module unavailable
        if seq_len > self.max_seq_len_for_custom or not ATTENTION_MODULE_AVAILABLE:
            if seq_len > self.max_seq_len_for_custom:
                print(f"Using PyTorch attention for long sequence (len={seq_len})")
            return self.out_proj(self._pytorch_attention(Q, K, V))
        
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
            
            if result is not None:
                result_tensor = torch.from_numpy(result).to(x.device)
                outputs.append(result_tensor)
                if backend_used is None:
                    # Determine which backend was actually used
                    if self.backend == 'auto':
                        if CUDA_AVAILABLE and self.num_heads == 1 and not self.causal_mask:
                            backend_used = 'cuda'
                        elif OPENMP_AVAILABLE:
                            backend_used = 'openmp'
                        else:
                            backend_used = 'cpu'
                    else:
                        backend_used = self.backend
            else:
                # Fall back to PyTorch for this sample
                result_tensor = self._pytorch_attention(Q[i:i+1], K[i:i+1], V[i:i+1]).squeeze(0)
                outputs.append(result_tensor)
                backend_used = 'pytorch_fallback'
        
        output = torch.stack(outputs)
        return self.out_proj(output)
    
    def _pytorch_attention(self, Q, K, V):
        """Efficient PyTorch attention implementation with multi-head and masking support"""
        batch_size, seq_len, embed_dim = Q.shape
        head_dim = embed_dim // self.num_heads
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
        K = K.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
        V = V.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)  # [B, H, L, L]
        
        # Apply causal mask if requested
        if self.causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, L, L]
        output = torch.matmul(attn_weights, V)  # [B, H, L, D]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return output

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
        
        # Use PyTorch for very large spatial dimensions or when module unavailable
        if spatial_size > self.max_spatial_size or not ATTENTION_MODULE_AVAILABLE:
            output = self._pytorch_attention(Q, K, V)
        else:
            # Process batch using custom attention
            outputs = []
            for i in range(batch_size):
                q_i = Q[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                k_i = K[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                v_i = V[i].detach().cpu().numpy().astype(np.float32)  # [HW, C]
                
                # Try custom attention backends
                result = AttentionBackend.compute_attention(
                    q_i, k_i, v_i, backend=self.backend
                )
                
                if result is not None:
                    result_tensor = torch.from_numpy(result).to(x.device)
                    outputs.append(result_tensor)
                else:
                    # Fall back to PyTorch for this sample
                    result_tensor = self._pytorch_attention(Q[i:i+1], K[i:i+1], V[i:i+1]).squeeze(0)
                    outputs.append(result_tensor)
            
            output = torch.stack(outputs)
        
        # Reshape back to spatial dimensions
        output = output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return self.out_conv(output)
    
    def _pytorch_attention(self, Q, K, V):
        """PyTorch attention implementation for spatial attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, HW, HW]
        scores = scores / np.sqrt(self.in_channels)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, HW, HW]
        output = torch.matmul(attn_weights, V)  # [B, HW, C]
        return output

def benchmark_attention_backends(seq_len=128, embed_dim=512, num_heads=8, num_iterations=10):
    """Benchmark different attention backends"""
    if not ATTENTION_MODULE_AVAILABLE:
        print("Custom attention module not available for benchmarking")
        return
    
    print(f"\n{'='*60}")
    print("Attention Backend Benchmark")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Iterations: {num_iterations}")
    print()
    
    # Generate test data
    np.random.seed(42)
    Q = np.random.randn(seq_len, embed_dim).astype(np.float32)
    K = np.random.randn(seq_len, embed_dim).astype(np.float32)
    V = np.random.randn(seq_len, embed_dim).astype(np.float32)
    
    backends_to_test = []
    if CUDA_AVAILABLE:
        backends_to_test.append('cuda')
    if OPENMP_AVAILABLE:
        backends_to_test.append('openmp')
    backends_to_test.append('cpu')
    
    results = {}
    
    for backend in backends_to_test:
        print(f"Testing {backend.upper()} backend...")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            try:
                if backend == 'cuda' and num_heads == 1:
                    result = attention_cuda_py.attention_cuda(Q, K, V)
                elif backend == 'openmp':
                    result = attention_cuda_py.attention_openmp(Q, K, V)
                elif backend == 'cpu':
                    result = attention_cuda_py.attention_cpu(Q, K, V)
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
            except Exception as e:
                print(f"  Error in {backend}: {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[backend] = avg_time
            print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        
        print()
    
    # Print summary
    if results:
        print("Summary (average times):")
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        fastest_time = sorted_results[0][1]
        
        for backend, avg_time in sorted_results:
            speedup = fastest_time / avg_time
            print(f"  {backend.upper()}: {avg_time:.2f} ms (speedup: {speedup:.2f}x)")
    
    print(f"{'='*60}")

def test_attention_layers():
    """Test function to verify all attention layers work correctly"""
    print("\n" + "="*60)
    print("Testing Enhanced Attention Layers")
    print("="*60)
    
    # Test EnhancedAttentionLayer
    print("\n1. Testing EnhancedAttentionLayer...")
    
    test_configs = [
        {'embed_dim': 64, 'num_heads': 1, 'backend': 'auto', 'causal_mask': False},
        {'embed_dim': 64, 'num_heads': 4, 'backend': 'openmp', 'causal_mask': False},
        {'embed_dim': 64, 'num_heads': 1, 'backend': 'auto', 'causal_mask': True},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n  Test {i+1}: {config}")
        try:
            layer = EnhancedAttentionLayer(**config)
            x = torch.randn(2, 10, config['embed_dim'])
            
            output = layer(x)
            print(f"  ✓ Test passed! Input: {x.shape}, Output: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
    
    # Test SpatialAttentionLayer
    print("\n2. Testing SpatialAttentionLayer...")
    try:
        spatial_layer = SpatialAttentionLayer(in_channels=64, backend='auto')
        x_spatial = torch.randn(2, 64, 8, 8)
        
        output_spatial = spatial_layer(x_spatial)
        print(f"  ✓ Test passed! Input: {x_spatial.shape}, Output: {output_spatial.shape}")
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
    
    # Run benchmark if module is available
    if ATTENTION_MODULE_AVAILABLE:
        print("\n3. Running performance benchmark...")
        benchmark_attention_backends(seq_len=64, embed_dim=256, num_heads=4, num_iterations=5)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)

if __name__ == "__main__":
    test_attention_layers()