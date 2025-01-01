# Install required packages
pip install gputil
pip install tensorboard
pip install torch-lr-finder

# 1. Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import multiprocessing 
import torch.cuda.amp as amp
import gc
import psutil

# Function to get GPU memory info using PyTorch
def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2  # MB
        }
    return None 

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}") 