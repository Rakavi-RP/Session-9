# LR Finder Cell with multiple runs
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import DataLoader

def run_lr_finder(num_runs=5, num_iter=200):
    suggested_lrs = []
    
    for run in range(num_runs):
        print(f"\nLR Finder Run {run + 1}/{num_runs}")
        
        # Setup model
        model = ResNet50(num_classes=1000).to('cuda')
        
        # Setup data
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        train_dir = '/mnt/imagenet/home/ubuntu/imagenet/ILSVRC/Data/CLS-LOC/train'
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.workers,
            pin_memory=True
        )
        
        optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        amp_config = {'device_type': 'cuda', 'dtype': torch.float16}
        grad_scaler = torch.cuda.amp.GradScaler()
        
        lr_finder = LRFinder(
            model,
            optimizer,
            criterion,
            device='cuda',
            amp_backend='torch',
            amp_config=amp_config,
            grad_scaler=grad_scaler
        )
        
        print("Running LR range test...")
        lr_finder.range_test(train_loader, start_lr=1e-5, end_lr=1.0, num_iter=num_iter)
        
        plt.figure(figsize=(10, 6))
        lr_finder.plot()
        plt.grid(True)
        plt.show()
        
        # Find the point before loss starts increasing dramatically
        loss_values = lr_finder.history['loss']
        lr_values = lr_finder.history['lr']
        
        min_loss_idx = np.argmin(loss_values)
        best_lr = lr_values[min_loss_idx]
        
        suggested_lr = best_lr * 0.05  # Conservative scaling
        suggested_lrs.append(suggested_lr)
        
        print(f"Run {run + 1} - Best LR: {best_lr:.2E}, Suggested LR: {suggested_lr:.2E}")
        
        lr_finder.reset()
        del model, optimizer, lr_finder
        torch.cuda.empty_cache()
    
    # Remove outliers before calculating statistics
    suggested_lrs = np.array(suggested_lrs)
    median = np.median(suggested_lrs)
    mad = np.median(np.abs(suggested_lrs - median))
    modified_z_scores = 0.6745 * (suggested_lrs - median) / mad
    suggested_lrs = suggested_lrs[np.abs(modified_z_scores) < 3.5]
    
    print("\nFinal Results (after removing outliers):")
    print(f"All suggested LRs: {[f'{lr:.2E}' for lr in suggested_lrs]}")
    print(f"Median LR: {np.median(suggested_lrs):.2E}")
    print(f"Mean LR: {np.mean(suggested_lrs):.2E}")
    print(f"Std Dev: {np.std(suggested_lrs):.2E}")
    
    params.base_lr = np.median(suggested_lrs)
    print(f"\nUpdated params.base_lr to: {params.base_lr:.2E}")

# Run multiple iterations
run_lr_finder(num_runs=5, num_iter=200)