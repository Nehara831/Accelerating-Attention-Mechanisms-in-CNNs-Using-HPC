import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from src.python.attention_cnn import AttentionCNN

def train_model(subset_size=1000, device_type='cuda', num_threads=None, batch_size=32):
    """
    Train the model with specified device type
    Args:
        subset_size: Number of samples to use for training
        device_type: 'cuda' or 'cpu'
        num_threads: Number of CPU threads to use (only for CPU mode)
        batch_size: Batch size for training
    """
    print("DEBUG: Starting training setup")  # Debug print
    
    # Set device and threads
    if device_type == 'cpu':
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        device = torch.device("cpu")
        print(f"Running on CPU with {num_threads if num_threads else 'default'} threads")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set CUDA memory management
        torch.cuda.empty_cache()
        print(f"Running on {device}")
    
    print("DEBUG: Loading dataset")  # Debug print
    
    # Data loading with single channel normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 like CIFAR-10
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std for single channel
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 download=True, transform=transform)
    
    # Create a subset sampler
    indices = list(range(len(train_dataset)))
    subset_indices = indices[:subset_size]
    sampler = SubsetRandomSampler(subset_indices)
    
    # Set number of workers based on device type
    num_workers = 0 if device_type == 'cpu' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=(device_type == 'cuda'))  # Enable pin_memory for GPU
    
    print(f"Using {subset_size} samples for training with batch size {batch_size}")
    
    print("DEBUG: Creating model")  # Debug print
    
    # Model setup with single channel input
    model = AttentionCNN(num_classes=10, in_channels=1, use_cuda_attention=(device_type == 'cuda'))
    model.to(device)
    
    # Enable cudnn benchmarking for faster training
    if device_type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("DEBUG: Starting training loop")  # Debug print
    
    # Training loop
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"DEBUG: Processing batch {i+1}")  # Debug print
            
            # Move data to device asynchronously
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if i == 0:  # Print shape only for first batch
                print(f"Input tensor shape: {inputs.shape}")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
                
                # Print GPU memory usage
                if device_type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                          f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Attention CNN')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--num-threads', type=int, default=None,
                      help='Number of CPU threads to use (only for CPU mode)')
    parser.add_argument('--subset-size', type=int, default=1000,
                      help='Number of samples to use for training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    args = parser.parse_args()
    
    train_model(subset_size=args.subset_size, 
               device_type=args.device,
               num_threads=args.num_threads,
               batch_size=args.batch_size)