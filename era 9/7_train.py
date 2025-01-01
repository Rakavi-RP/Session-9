# 7. Training function 
def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, 
                       desc=f"Epoch {epoch}/{num_epochs} Training",
                       ncols=200,
                       leave=True)
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        # First unscale gradients and optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if total % 5000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update progress bar with fixed-width format
        current_lr = optimizer.param_groups[0]['lr']
        lr_str = f"{current_lr:8.2e}".ljust(10)  # Force fixed width
        
        progress_bar.set_description(
            f"Epoch {epoch}/{num_epochs} Training | "
            f"loss: {running_loss/total:.3f} | "
            f"acc: {100.*correct/total:.2f}% | "
            f"lr: {lr_str}"
        )
    
    return 100. * correct / total, running_loss / len(train_loader) 