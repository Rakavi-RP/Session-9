# 6. Checkpoint functions
def save_checkpoint(state, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
    torch.save(state, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

def save_best_model(state, is_best, checkpoint_dir):
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f"\nBest model saved: {best_path}")

def load_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resumed from epoch {start_epoch} with best accuracy: {best_accuracy:.2f}%")
        return start_epoch, best_accuracy
    return 0, 0.0 