import os
import sys

os.environ['OMP_NUM_THREADS'] = '16'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from src.python.attention_cnn import AttentionCNN
import time
import random
import numpy as np

def set_optimal_openmp_threads():
    """Set optimal OpenMP thread count"""
    optimal_threads = 16
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    print(f"Set optimal OpenMP threads to: {os.environ['OMP_NUM_THREADS']}")

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to: {seed}")

def train_model(subset_size=1000, device_type='cuda', attention_backend='auto', 
                num_threads=None, batch_size=32, seed=42):
    """Train model with optimal OpenMP settings"""
    
    
    # Set seed for reproducibility
    set_seed(seed)
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Random seed: {seed}")
    print(f"PyTorch device: {device_type}")
    print(f"Attention backend: {attention_backend}")
    
    print(f"Subset size: {subset_size}")
    print(f"Batch size: {batch_size}")
    
    # Set device and threads
    if device_type == 'cpu':
        if num_threads is not None:
            torch.set_num_threads(16)
            print(f"PyTorch CPU threads: {num_threads}")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        else:
            print("CUDA not available, falling back to CPU")
    
    print("="*60)
    
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
    
    # Set number of workers based on device type
    num_workers = 0 if device_type == 'cpu' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=(device_type == 'cuda'))  
    
    print(f"Dataset loaded: {subset_size} samples, {len(train_loader)} batches")
    
    # Create model with explicit backend selection
    print(f"\nCreating model with attention backend: {attention_backend}")
    model = AttentionCNN(num_classes=10, in_channels=1, backend=attention_backend)
    model.to(device)
    
    # Re-disable benchmark after model creation if deterministic mode is required
    if seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device_type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)

    # Start timing
    total_start_time = time.time()

    # Training loop
    model.train()
    for epoch in range(9):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            if i == 0 and epoch == 0:
                print(f"Input tensor shape: {inputs.shape}")
                print(f"Processing first batch...")
            
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
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
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {avg_loss:.4f}, Accuracy: {batch_accuracy:.2f}%')
                running_loss = 0.0
                running_correct = 0
                total_samples = 0
                
                if device_type == 'cuda':
                    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                          f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
        
        # Calculate epoch accuracy
        epoch_accuracy = 100.0 * running_correct / total_samples if total_samples > 0 else 0
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")
        if total_samples > 0:
            print(f"Epoch {epoch+1} final accuracy: {epoch_accuracy:.2f}%")

    # End timing
    total_end_time = time.time()
    elapsed = total_end_time - total_start_time
    
    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total training time: {elapsed:.2f} seconds")
    print(f"PyTorch device: {device_type.upper()}")
    print(f"Attention backend: {attention_backend.upper()}")
    print(f"Random seed used: {seed}")
    print(f"Average time per epoch: {elapsed/9:.2f} seconds")
    
    if attention_backend == 'openmp':
        print(f"OpenMP threads used: {os.environ.get('OMP_NUM_THREADS', 'default')}")
    
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Attention CNN with Optimal Settings')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                      help='Device to use for PyTorch operations (cuda or cpu)')
    
    # Attention backend selection
    parser.add_argument('--attention-backend', type=str, default='auto', 
                      choices=['auto', 'cuda', 'openmp', 'cpu', 'pytorch'],
                      help='Backend for attention computation')
    
    # Training parameters
    parser.add_argument('--num-threads', type=int, default=None,
                      help='Number of CPU threads to use (only for CPU mode)')
    parser.add_argument('--subset-size', type=int, default=1000,
                      help='Number of samples to use for training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    train_model(subset_size=args.subset_size, 
               device_type=args.device,
               attention_backend=args.attention_backend,
               num_threads=args.num_threads,
               batch_size=args.batch_size,
               seed=args.seed)