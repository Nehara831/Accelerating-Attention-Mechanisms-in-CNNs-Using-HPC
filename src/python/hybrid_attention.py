import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import SpatialAttentionLayer, EnhancedAttentionLayer



class TrueHybridAttentionCNN(nn.Module):
    
    def __init__(self, num_classes=10, in_channels=1):
        super(TrueHybridAttentionCNN, self).__init__()
        

        
        # Import attention layers
        from src.python.attention_layer import SpatialAttentionLayer, EnhancedAttentionLayer
        
        # CNN layers 
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.spatial_attn1_cpu = SpatialAttentionLayer(128, backend='openmp')
        self.spatial_attn2_cpu = SpatialAttentionLayer(256, backend='openmp')
        self.spatial_attn3_cpu = SpatialAttentionLayer(512, backend='openmp')
        
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
        super(TrueHybridAttentionCNN, self).to(device)
        
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
        
        # Global attention on GPU 
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
            x_attended = attention_layer(x_cpu)
        
        x_attended = x_attended.to(original_device)
        x_attended.requires_grad_(x.requires_grad)
        
        return x_attended
    
    def _apply_global_attention_gpu(self, x, original_device):
        """Apply global attention on GPU using CUDA"""
        if torch.cuda.is_available() and original_device.type == 'cuda':
            return self.global_attention(x)
        else:
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                self.global_attention.cuda()
                x_attended = self.global_attention(x_gpu)
                return x_attended.to(original_device)
            else:
                return self.global_attention(x)

class SwappedAttentionCNN(nn.Module):
    
    def __init__(self, num_classes=10, in_channels=1):
        
        super(SwappedAttentionCNN, self).__init__()
        
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Spatial attention layers 
        self.spatial_attn1 = SpatialAttentionLayer(128, backend='cuda')
        self.spatial_attn2 = SpatialAttentionLayer(256, backend='cuda')
        self.spatial_attn3 = SpatialAttentionLayer(512, backend='cuda')
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling 
        self.global_attention = EnhancedAttentionLayer(512, num_heads=1, backend='openmp')
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block with spatial attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.spatial_attn1(x)  
        x = self.pool(x)
        
        # Third conv block with spatial attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.spatial_attn2(x)  
        x = self.pool(x)
        
        # Fourth conv block with spatial attention
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.spatial_attn3(x)  
        x = self.pool(x)
        
        # Global enhanced attention processing:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)  
        x = self.global_attention(x)  
        x = x.squeeze(1) 
        
        # Classification
        x = self.classifier(x)
        return x


   