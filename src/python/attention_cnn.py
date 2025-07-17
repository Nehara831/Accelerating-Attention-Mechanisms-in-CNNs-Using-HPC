import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import SpatialAttentionLayer,EnhancedAttentionLayer

class AttentionCNN(nn.Module):
    """CNN with integrated spatial attention mechanisms with explicit backend control"""
    
    def __init__(self, num_classes=10, in_channels=1, backend='auto'):
        
        super(AttentionCNN, self).__init__()
        
        self.backend = backend
        print(f"AttentionCNN initialized with backend: {backend}")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Spatial attention layers with explicit backend control
        self.spatial_attn1 = SpatialAttentionLayer(128, backend=backend)
        self.spatial_attn2 = SpatialAttentionLayer(256, backend=backend)
        self.spatial_attn3 = SpatialAttentionLayer(512, backend=backend)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling 
        self.global_attention = EnhancedAttentionLayer(512, num_heads=8, backend=backend)
        
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
        x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, 512]
        x = self.global_attention(x)  # Apply enhanced attention
        x = x.squeeze(1)  # Remove sequence dimension [batch_size, 512]
        
        # Classification
        x = self.classifier(x)
        return x