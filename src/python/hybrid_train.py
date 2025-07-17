#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.python.attention_cnn import AttentionCNN
from src.python.hybrid_attention import SwappedHybridAttentionCNN,TrueHybridAttentionCNN
from torchvision import datasets, transforms
import time
import random
import numpy as np

# Set optimal OpenMP settings for hybrid processing
os.environ['OMP_NUM_THREADS'] = '16'


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(f"Random seeds set to: {seed}")



def create_hybrid_model(configuration='swapped', num_classes=10, in_channels=1):
    """Create hybrid model with different configurations"""
    
    
    if configuration == 'swapped':
        return SwappedHybridAttentionCNN(num_classes, in_channels)
    
    elif configuration == 'original':
        return TrueHybridAttentionCNN(num_classes, in_channels) 
    
    elif configuration == 'cuda_only':
        return AttentionCNN(num_classes, in_channels, backend='cuda')
    
    elif configuration == 'openmp_only':
        return AttentionCNN(num_classes, in_channels, backend='openmp')
    
    else:
        return SwappedHybridAttentionCNN(num_classes, in_channels)

def train_swapped_hybrid_model(subset_size=1000, device_type='cuda', batch_size=32, 
                              seed=42, configuration='swapped'):
    """Train model with swapped hybrid approach"""
    
    set_seed(seed)

    
    # Optimize PyTorch for hybrid processing
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and device_type == 'cuda' else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    else:
        print(f"🖥️  Using CPU device")
    
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                    download=True, transform=transform)
    
    indices = list(range(len(train_dataset)))
    subset_indices = indices[:subset_size]
    sampler = SubsetRandomSampler(subset_indices)
    
    num_workers = 0 if device_type == 'cpu' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=(device_type == 'cuda'))  
    
    
    # Create hybrid model
    model = create_hybrid_model(configuration=configuration)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    # Training
    total_start_time = time.time()
    model.train()
    
    for epoch in range(9):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for i, (inputs, labels) in enumerate(train_loader):

            
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass with hybrid processing
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                running_correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                
                if i % 10 == 9:
                    batch_accuracy = 100.0 * running_correct / total_samples
                    avg_loss = running_loss / 10
                    print(f' Epoch {epoch+1}, Batch {i+1}, Loss: {avg_loss:.4f}, Accuracy: {batch_accuracy:.2f}%')
                    running_loss = 0.0
                    running_correct = 0
                    total_samples = 0
                    
                                            
            except Exception as e:
                print(f" Error in batch {i}: {e}")
                print(f"Skipping batch and continuing...")
                continue
        
        # Epoch summary
        epoch_accuracy = 100.0 * running_correct / total_samples if total_samples > 0 else 0
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f"⏱  Epoch {epoch+1}: {epoch_time:.2f}s")
        if total_samples > 0:
            print(f" Epoch {epoch+1} accuracy: {epoch_accuracy:.2f}%")

    # Training completed
    total_end_time = time.time()
    elapsed = total_end_time - total_start_time
    
    print("="*60)
    print("🏁 TRAINING COMPLETED")
    print("="*60)
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f" Avg per epoch: {elapsed/9:.2f} seconds")
    
    
    
    
    return elapsed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='🔄 Swapped Hybrid Training: Spatial→CUDA, Global→OpenMP')
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                      help='Device for PyTorch operations')
    parser.add_argument('--subset-size', type=int, default=1000,
                      help='Number of training samples')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--config', type=str, default='swapped',
                      choices=['swapped', 'original', 'cuda_only', 'openmp_only'],
                      help='Hybrid configuration')
    parser.add_argument('--compare', action='store_true',
                      help='Compare all configurations')
    
    args = parser.parse_args()
    

        # Run training with specified configuration
    train_swapped_hybrid_model(
        subset_size=args.subset_size,
        device_type=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        configuration=args.config
    )