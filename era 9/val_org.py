# Cell to organize validation images into class folders
import os
import shutil
from tqdm import tqdm

def organize_val_directory():
    # Paths
    val_dir = '/mnt/imagenet/home/ubuntu/imagenet/ILSVRC/Data/CLS-LOC/val'
    val_info_file = '/mnt/imagenet/home/ubuntu/imagenet/LOC_val_solution.csv'
    
    # Check if validation is already organized
    sample_files = os.listdir(val_dir)[:5]
    if all(not f.endswith('.JPEG') for f in sample_files):
        print("Validation directory appears to be already organized!")
        return
    
    # Create a mapping from image to class
    print("Reading validation info...")
    img_to_class = {}
    with open(val_info_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            img_name, class_id = line.strip().split(',')
            class_id = class_id.split()[0]  # Get first word (class ID)
            img_to_class[img_name] = class_id
    
    # Create class directories and move files directly
    print("Organizing validation images...")
    for img_name, class_id in tqdm(img_to_class.items()):
        # Create class directory
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        # Move image to class directory
        src = os.path.join(val_dir, img_name)
        if os.path.exists(src):
            dst = os.path.join(class_dir, img_name)
            try:
                shutil.move(src, dst)
            except:
                continue
    
    print("Validation directory organized!")
    
# Run the organization
organize_val_directory()