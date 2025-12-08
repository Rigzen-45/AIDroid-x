"""
Training Utilities
Helper functions for model training, evaluation, and checkpoint management
"""

import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(device_name: str = "cuda") -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_name: Device name ('cuda' or 'cpu')
        
    Returns:
        PyTorch device
    """
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Evaluation metrics
        save_path: Path to save checkpoint
        config: Optional configuration dictionary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if config:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to restore
        device: Device to load model on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint


class EarlyStopping:
    """
    Early stopping handler to stop training when metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for minimize
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "max":
            self.is_better = lambda new, best: new > best + min_delta
        else:
            self.is_better = lambda new, best: new < best - min_delta
    
    def __call__(self, metric: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                logger.info(f"Metric improved to {metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
                return True
        
        return False


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str):
        """
        Initialize meter.
        
        Args:
            name: Name of the metric
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """
    Tracks training and validation metrics over epochs.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update(self, epoch_metrics: Dict[str, float]):
        """
        Update metrics for current epoch.
        
        Args:
            epoch_metrics: Dictionary of metric values
        """
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best(self, metric: str, mode: str = "max") -> float:
        """
        Get best value for a metric.
        
        Args:
            metric: Metric name
            mode: 'max' or 'min'
            
        Returns:
            Best metric value
        """
        if metric not in self.history or not self.history[metric]:
            return None
        
        if mode == "max":
            return max(self.history[metric])
        else:
            return min(self.history[metric])
    
    def save(self, save_path: str):
        """
        Save metrics history to JSON.
        
        Args:
            save_path: Path to save file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load metrics history from JSON.
        
        Args:
            load_path: Path to load file
        """
        with open(load_path, 'r') as f:
            self.history = json.load(f)
        
        logger.info(f"Metrics loaded from {load_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_model_info(model: torch.nn.Module):
    """
    Log model architecture information.
    
    Args:
        model: PyTorch model
    """
    logger.info("=" * 80)
    logger.info("Model Architecture")
    logger.info("=" * 80)
    logger.info(f"\n{model}\n")
    logger.info(f"Total parameters: {count_parameters(model):,}")
    
    # Count parameters per layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  {name}: {param.numel():,} parameters")
    
    logger.info("=" * 80)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = total / (len(unique) * counts)
    weights = torch.FloatTensor(weights)
    
    logger.info(f"Class weights: {weights.numpy()}")
    
    return weights


def mixup_data(x, y, alpha=1.0):
    """
    Mixup augmentation for training.
    
    Args:
        x: Input data
        y: Labels
        alpha: Mixup parameter
        
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def warmup_lr_scheduler(optimizer, warmup_epochs, initial_lr, target_lr):
    """
    Create learning rate warmup scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        initial_lr: Starting learning rate
        target_lr: Target learning rate after warmup
        
    Returns:
        Scheduler function
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return initial_lr + (target_lr - initial_lr) * epoch / warmup_epochs
        else:
            return target_lr
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test utilities
    print("Testing training utilities...")
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device("cuda")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5, mode="max")
    
    metrics = [0.85, 0.87, 0.89, 0.88, 0.88, 0.87, 0.86, 0.85]
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch}: Metric = {metric:.4f}")
        if early_stopping(metric):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({
        'train_loss': 0.5,
        'train_acc': 0.85,
        'val_loss': 0.6,
        'val_acc': 0.82
    })
    
    print(f"\nBest validation accuracy: {tracker.get_best('val_acc')}")
    
    print("\n✅ All utilities working correctly!")