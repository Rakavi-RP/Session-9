# 10. Main training loop
if __name__ == '__main__':
    params = Params()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories in home folder
    home_dir = os.path.expanduser('~')  # This gets your home directory
    checkpoint_dir = os.path.join(home_dir, 'checkpoints', params.name)
    runs_dir = os.path.join(home_dir, 'runs')
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    # Setup tensorboard with home path
    writer = SummaryWriter(os.path.join(runs_dir, params.name))
    
    # Setup data
    train_loader, validation_loader, num_classes = setup_data('/mnt/imagenet/home/ubuntu/imagenet/ILSVRC/Data/CLS-LOC', params)
    
    # Create model
    model = ResNet50(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    
    # Proper initialization of GradScaler
    scaler = torch.cuda.amp.GradScaler(
        enabled=True,
        init_scale=2**16,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    # Implement learning rate warmup
    def get_lr(epoch):
        if epoch < params.warmup_epochs:
            return params.base_lr * (epoch + 1) / params.warmup_epochs
        return params.base_lr
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=params.base_lr/25,  # Initial learning rate (max_lr/div_factor)
        momentum=0.85,  # Starting momentum for OneCycleLR
        weight_decay=params.weight_decay,
        nesterov=True
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * params.num_epochs
    
    # OneCycleLR scheduler
    #scheduler = optim.lr_scheduler.OneCycleLR(
        #optimizer,
        #max_lr=params.base_lr * 0.8,  #reducing lr to stabilize
        #total_steps=total_steps,
        #pct_start=0.3,   #changed from 0.2
        #anneal_strategy='cos',
        #cycle_momentum=True,
        #base_momentum=0.85,
        #max_momentum=0.95,
        #div_factor=25.0,    #changed from 20.0
        #final_div_factor=10000.0   #changed from 1000.0
    #)

    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # We want to maximize accuracy
        factor=0.5,          # Reduce LR by half when plateauing
        patience=3,          # Wait 3 epochs before reducing
        verbose=True         # Print when LR changes
    )
    
    # Load checkpoint if exists
    start_epoch, best_accuracy = load_checkpoint(checkpoint_dir, model, optimizer, scheduler)
    
    # Training loop
    for epoch in range(start_epoch, params.num_epochs):
        print(f"\nEpoch {epoch + 1}/{params.num_epochs}")
        
        train_accuracy, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            scheduler, scaler, device, epoch + 1, params.num_epochs
        )
        
        val_accuracy, val_loss = validate(
            model, validation_loader, criterion, 
            device, epoch + 1
        )
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Add scaler statistics logging
        writer.add_scalar('Grad_scale', scaler.get_scale(), epoch)
        
        # Log memory usage using PyTorch instead of GPUtil
        gpu_memory = get_gpu_memory()
        if gpu_memory:
            writer.add_scalar('GPU_Memory_Allocated_MB', gpu_memory['allocated'], epoch)
            writer.add_scalar('GPU_Memory_Cached_MB', gpu_memory['cached'], epoch)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(val_accuracy, best_accuracy)
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': best_accuracy,
        }
        
        save_checkpoint(checkpoint_state, checkpoint_dir)
        save_best_model(checkpoint_state, is_best, checkpoint_dir)
        
        print(f"\nResults - Train acc: {train_accuracy:.2f}%, Val acc: {val_accuracy:.2f}%")
        
        scheduler.step(val_accuracy)  # Update based on validation accuracy
        
    
    writer.close() 