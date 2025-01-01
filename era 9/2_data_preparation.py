# 2. Data preparation
import os
from pathlib import Path

def prepare_data():
    # Define paths to your mounted ImageNet
    base_dir = '/mnt/imagenet/home/ubuntu/imagenet/ILSVRC/Data/CLS-LOC'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # Print paths for verification
    print(f"Checking directories:")
    print(f"Base dir: {base_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    
    # Verify the paths
    assert os.path.exists(train_dir), f"Training directory not found at {train_dir}"
    assert os.path.exists(val_dir), f"Validation directory not found at {val_dir}"
    
    # Print statistics
    n_train_classes = len(os.listdir(train_dir))
    n_val_images = len(os.listdir(val_dir))
    print(f"\nFound {n_train_classes} training classes")
    print(f"Found {n_val_images} validation images")
    
    return base_dir

# Get the data directory
data_dir = prepare_data()