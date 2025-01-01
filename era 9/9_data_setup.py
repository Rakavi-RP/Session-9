# 9. Data setup
def setup_data(data_dir, params):
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    validation_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.workers,
        pin_memory=True,
        drop_last=True
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.workers,
        pin_memory=True
    )
    
    return train_loader, validation_loader, len(train_dataset.classes) 