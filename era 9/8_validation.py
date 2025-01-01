# 8. Updated Validation function
def validate(model, validation_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(validation_loader,
                       desc=f"Validating",
                       ncols=100,
                       leave=True)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Fixed autocast syntax
            with torch.autocast(device_type='cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': f"{running_loss/total:.3f}",
                'accuracy': f"{100.*correct/total:.2f}%"
            })
    
    return 100. * correct / total, running_loss / len(validation_loader) 