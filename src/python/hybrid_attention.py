import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import SpatialAttentionLayer, EnhancedAttentionLayer

class HybridAttentionCNN(nn.Module):
    
    
    def __init__(self, num_classes=10, in_channels=1, device_type='cuda'):
        
        super(HybridAttentionCNN, self).__init__()
        
        self.device_type = device_type
        print(f"HybridAttentionCNN initialized:")
        print(f"  PyTorch device: {device_type}")
        print(f"  Spatial Attention: OpenMP (CPU)")
        print(f"  Global Attention: CUDA (GPU)")
        
        # Convolutional layers (PyTorch - will run on specified device)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Spatial attention layers - FORCE OpenMP (CPU)
        self.spatial_attn1 = SpatialAttentionLayer(128, backend='openmp')
        self.spatial_attn2 = SpatialAttentionLayer(256, backend='openmp')
        self.spatial_attn3 = SpatialAttentionLayer(512, backend='openmp')
        
        # Pooling and normalization (PyTorch)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global attention - FORCE CUDA (GPU)
        self.global_attention = EnhancedAttentionLayer(512, num_heads=1, backend='cuda')
        
        # Classification head (PyTorch)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN operations on specified device (GPU/CPU)
        device = x.device
        
        # First conv block (PyTorch)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block + spatial attention (OpenMP on CPU)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self._apply_spatial_attention(x, self.spatial_attn1, device)
        x = self.pool(x)
        
        # Third conv block + spatial attention (OpenMP on CPU)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self._apply_spatial_attention(x, self.spatial_attn2, device)
        x = self.pool(x)
        
        # Fourth conv block + spatial attention (OpenMP on CPU)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self._apply_spatial_attention(x, self.spatial_attn3, device)
        x = self.pool(x)
        
        # Global pooling and prepare for global attention
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), -1)
        
        # Global attention (CUDA on GPU)
        x = x.unsqueeze(1)
        x = self._apply_global_attention(x, device)
        x = x.squeeze(1)
        
        # Classification (PyTorch)
        x = self.classifier(x)
        return x
    
    def _apply_spatial_attention(self, x, attention_layer, original_device):
        
        # Move to CPU for OpenMP processing
        x_cpu = x.detach().cpu()
        
        # Apply OpenMP spatial attention on CPU
        x_attended = attention_layer(x_cpu)
        
        # Move back to original device
        return x_attended.to(original_device)
    
    def _apply_global_attention(self, x, original_device):
        
        # For CUDA global attention, we need to handle device transfers
        if self.device_type == 'cuda' and torch.cuda.is_available():
            # Move to GPU for CUDA processing if not already there
            if x.device.type != 'cuda':
                x_gpu = x.cuda()
            else:
                x_gpu = x
            
            # Apply CUDA global attention on GPU
            x_attended = self.global_attention(x_gpu)
            
            # Move back to original device if needed
            return x_attended.to(original_device)
        else:
            # Fallback: apply on current device
            return self.global_attention(x)





class TrueHybridAttentionCNN(nn.Module):
    """CNN with true hybrid OpenMP + CUDA attention and proper device handling"""
    
    def __init__(self, num_classes=10, in_channels=1):
        super(TrueHybridAttentionCNN, self).__init__()
        
        print("🔄 True Hybrid Model:")
        print("  🧵 Spatial Attention → OpenMP (CPU)")
        print("  🚀 Global Attention → CUDA (GPU)")
        print("  🏗️  CNN Layers → PyTorch (GPU)")
        
        # Import attention layers
        from src.python.attention_layer import SpatialAttentionLayer, EnhancedAttentionLayer
        
        # CNN layers (will run on GPU)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Create separate CPU-based spatial attention layers
        self.spatial_attn1_cpu = SpatialAttentionLayer(128, backend='openmp')
        self.spatial_attn2_cpu = SpatialAttentionLayer(256, backend='openmp')
        self.spatial_attn3_cpu = SpatialAttentionLayer(512, backend='openmp')
        
        # Keep these on CPU permanently
        self.spatial_attn1_cpu.cpu()
        self.spatial_attn2_cpu.cpu()
        self.spatial_attn3_cpu.cpu()
        
        # Pooling and normalization (GPU)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global attention - FORCE CUDA (GPU)
        self.global_attention = EnhancedAttentionLayer(512, num_heads=1, backend='cuda')
        
        # Classification head (GPU)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def to(self, device):
        """Override to ensure CPU layers stay on CPU"""
        # Move most layers to the specified device
        super(TrueHybridAttentionCNN, self).to(device)
        
        # But keep spatial attention layers on CPU
        self.spatial_attn1_cpu.cpu()
        self.spatial_attn2_cpu.cpu()
        self.spatial_attn3_cpu.cpu()
        
        return self
        
    def forward(self, x):
        device = x.device
        
        # CNN operations on GPU
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Spatial attention on CPU (OpenMP)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self._apply_spatial_attention_cpu(x, self.spatial_attn1_cpu, device)
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self._apply_spatial_attention_cpu(x, self.spatial_attn2_cpu, device)
        x = self.pool(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self._apply_spatial_attention_cpu(x, self.spatial_attn3_cpu, device)
        x = self.pool(x)
        
        # Global attention on GPU (CUDA)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self._apply_global_attention_gpu(x, device)
        x = x.squeeze(1)
        
        # Classification on GPU
        x = self.classifier(x)
        return x
    
    def _apply_spatial_attention_cpu(self, x, attention_layer, original_device):
        """Apply spatial attention on CPU using OpenMP"""
        # Transfer to CPU
        x_cpu = x.detach().cpu()
        
        # Ensure attention layer is on CPU
        attention_layer.cpu()
        
        # Apply OpenMP spatial attention
        with torch.no_grad():
            # Temporarily disable gradients for CPU computation
            x_attended = attention_layer(x_cpu)
        
        # Transfer back to original device and restore gradients
        x_attended = x_attended.to(original_device)
        x_attended.requires_grad_(x.requires_grad)
        
        return x_attended
    
    def _apply_global_attention_gpu(self, x, original_device):
        """Apply global attention on GPU using CUDA"""
        # Ensure we're on GPU for CUDA processing
        if torch.cuda.is_available() and original_device.type == 'cuda':
            # Already on GPU, apply CUDA attention
            return self.global_attention(x)
        else:
            # Move to GPU, apply CUDA attention, move back
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                self.global_attention.cuda()
                x_attended = self.global_attention(x_gpu)
                return x_attended.to(original_device)
            else:
                # No CUDA available, fallback to CPU
                return self.global_attention(x)

# Inline class definitions for self-contained script
class SwappedHybridAttentionCNN(nn.Module):
    """CNN with swapped hybrid: Spatial→CUDA, Global→OpenMP"""
    
    def __init__(self, num_classes=10, in_channels=1):
        super(SwappedHybridAttentionCNN, self).__init__()
        
        print("🔄 Swapped Hybrid Model:")
        print("  🚀 Spatial Attention → CUDA (GPU)")
        print("  🧵 Global Attention → OpenMP (CPU)")
        print("  🏗️  CNN Layers → PyTorch (GPU)")
        
        # Import attention layers
        from src.python.attention_layer import SpatialAttentionLayer, EnhancedAttentionLayer
        
        # CNN layers (will run on GPU)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Spatial attention layers - FORCE CUDA (GPU)
        self.spatial_attn1_gpu = SpatialAttentionLayer(128, backend='cuda')
        self.spatial_attn2_gpu = SpatialAttentionLayer(256, backend='cuda')
        self.spatial_attn3_gpu = SpatialAttentionLayer(512, backend='cuda')
        
        # Pooling and normalization (GPU)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global attention - FORCE OpenMP (CPU)
        self.global_attention_cpu = EnhancedAttentionLayer(512, num_heads=1, backend='openmp')
        
        # Keep global attention on CPU permanently
        self.global_attention_cpu.cpu()
        
        # Classification head (GPU)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def to(self, device):
        """Override to ensure CPU layers stay on CPU"""
        super(SwappedHybridAttentionCNN, self).to(device)
        self.global_attention_cpu.cpu()
        return self
        
    def forward(self, x):
        device = x.device
        
        # CNN operations on GPU
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Spatial attention on GPU (CUDA)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self._apply_spatial_attention_gpu(x, self.spatial_attn1_gpu)
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self._apply_spatial_attention_gpu(x, self.spatial_attn2_gpu)
        x = self.pool(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self._apply_spatial_attention_gpu(x, self.spatial_attn3_gpu)
        x = self.pool(x)
        
        # Global attention on CPU (OpenMP)
        # x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = x.reshape(x.size(0), -1)
        # x = x.unsqueeze(1)
        # x = self._apply_global_attention_cpu(x, self.global_attention_cpu, device)
        # x = x.squeeze(1)
        
        # Classification on GPU
        x = self.classifier(x)
        return x
    
    def _apply_spatial_attention_gpu(self, x, attention_layer):
        """Apply spatial attention on GPU using CUDA"""
        attention_layer.to(x.device)
        return attention_layer(x)
    
    def _apply_global_attention_cpu(self, x, attention_layer, original_device):
        """Apply global attention on CPU using OpenMP"""
        x_cpu = x.detach().cpu()
        attention_layer.cpu()
        
        with torch.no_grad():
            x_attended = attention_layer(x_cpu)
        
        x_attended = x_attended.to(original_device)
        x_attended.requires_grad_(x.requires_grad)
        return x_attended


class AttentionCNN(nn.Module):
    """Backward compatible AttentionCNN that uses hybrid approach"""
    
    def __init__(self, num_classes=10, in_channels=1, backend='hybrid_layer'):
        super(AttentionCNN, self).__init__()
        
        if backend == 'hybrid_layer':
            self.model = HybridAttentionCNN(num_classes, in_channels, 'cuda')
        elif backend == 'hybrid_adaptive':
            self.model = AdaptiveHybridAttentionCNN(num_classes, in_channels, 'cuda')
        elif backend == 'hybrid_pipeline':
            self.model = PipelineHybridAttentionCNN(num_classes, in_channels, 'cuda')
        else:
            # Fallback to original implementation
            from .attention_layer import SpatialAttentionLayer
            self._init_original(num_classes, in_channels, backend)
    
    def _init_original(self, num_classes, in_channels, backend):
        """Original AttentionCNN implementation"""
        self.backend = backend
        print(f"AttentionCNN initialized with backend: {backend}")
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.spatial_attn1 = SpatialAttentionLayer(128, backend=backend)
        self.spatial_attn2 = SpatialAttentionLayer(256, backend=backend)
        self.spatial_attn3 = SpatialAttentionLayer(512, backend=backend)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.is_hybrid = False
    
    def forward(self, x):
        if hasattr(self, 'model'):
            return self.model(x)
        else:
            # Original forward pass
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.spatial_attn1(x)
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.spatial_attn2(x)
            x = self.pool(x)
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.spatial_attn3(x)
            x = self.pool(x)
            x = self.global_pool(x)
            x = x.reshape(x.size(0), -1)
            x = self.classifier(x)
            return x