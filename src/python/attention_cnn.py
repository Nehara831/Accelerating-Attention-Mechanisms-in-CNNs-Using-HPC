import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import CUDAAttentionLayer, SpatialAttentionLayer

class AttentionCNN(nn.Module):
    """CNN with integrated CUDA attention mechanisms"""
    
    def __init__(self, num_classes=10, in_channels=1, use_cuda_attention=True):
        super(AttentionCNN, self).__init__()
        
        # Convolutional layers - Fixed input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Attention layers
        self.spatial_attn1 = SpatialAttentionLayer(128, use_cuda_attention)
        self.spatial_attn2 = SpatialAttentionLayer(256, use_cuda_attention)
        self.spatial_attn3 = SpatialAttentionLayer(512, use_cuda_attention)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global attention on flattened features
        self.global_attention = CUDAAttentionLayer(512, num_heads=8, use_cuda=use_cuda_attention)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Print input shape for debugging
        # print(f"Input shape: {x.shape}")
        
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(f"After conv1+pool: {x.shape}")
        
        # Second conv block with spatial attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.spatial_attn1(x)  # Apply spatial attention
        x = self.pool(x)
        # print(f"After conv2+attn+pool: {x.shape}")
        
        # Third conv block with spatial attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.spatial_attn2(x)  # Apply spatial attention
        x = self.pool(x)
        # print(f"After conv3+attn+pool: {x.shape}")
        
        # Fourth conv block with spatial attention
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.spatial_attn3(x)  # Apply spatial attention
        x = self.pool(x)
        # print(f"After conv4+attn+pool: {x.shape}")
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # print(f"After adaptive_avg_pool2d: {x.shape}")
        
        # FIX: Use .reshape() instead of .view() for non-contiguous tensors
        x = x.reshape(x.size(0), -1)  # Changed from .view() to .reshape()
        # print(f"After reshape: {x.shape}")
        
        # Apply global attention to feature vector
        x = x.unsqueeze(1)  # Add sequence dimension [batch, 1, features]
        # print(f"Before global attention: {x.shape}")
        
        x = self.global_attention(x)
        # print(f"After global attention: {x.shape}")
        
        x = x.squeeze(1)  # Remove sequence dimension
        # print(f"After squeeze: {x.shape}")
        
        # Classification
        x = self.classifier(x)
        # print(f"Final output: {x.shape}")
        return x

class LightweightAttentionCNN(nn.Module):
    """Lightweight CNN with attention for mobile/edge deployment"""
    
    def __init__(self, num_classes=10, in_channels=1, use_cuda_attention=True):
        super(LightweightAttentionCNN, self).__init__()
        
        # Depthwise separable convolutions - Fixed input channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.dw_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pw_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        
        self.dw_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pw_conv3 = nn.Conv2d(64, 128, kernel_size=1)
        
        # Lightweight attention
        self.spatial_attn = SpatialAttentionLayer(128, use_cuda_attention)
        
        # Classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.pw_conv2(self.dw_conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.pw_conv3(self.dw_conv3(x)))
        x = self.spatial_attn(x)
        
        x = self.global_pool(x)
        # FIX: Use .reshape() instead of .view()
        x = x.reshape(x.size(0), -1)  # Changed from .view() to .reshape()
        x = self.classifier(x)
        return x