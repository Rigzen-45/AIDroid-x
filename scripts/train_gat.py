"""
Training Script for GAT Model
Paper params: 8 heads, 8 hidden units, LR=0.0005, 200 epochs
"""

import argparse
import yaml
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gat_model import GAT, GATTrainer
from src.dataset.graph_dataset import MalwareGraphDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train_gat.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict, logger):
    """Create train/val/test data loaders."""
    logger.info("Loading datasets...")
    
    # Load graphs
    train_graphs, train_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/train/graphs.pkl'
    )
    val_graphs, val_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/val/graphs.pkl'
    )
    test_graphs, test_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/test/graphs.pkl'
    )
    
    # Create datasets
    train_dataset = MalwareGraphDataset(
        root=config['paths']['graphs'] + '/train',
        graphs=train_graphs,
        labels=train_labels
    )
    
    val_dataset = MalwareGraphDataset(
        root=config['paths']['graphs'] + '/val',
        graphs=val_graphs,
        labels=val_labels
    )
    
    test_dataset = MalwareGraphDataset(
        root=config['paths']['graphs'] + '/test',
        graphs=test_graphs,
        labels=test_labels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['gat']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['gat']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['gat']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val: {len(val_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, train_dataset.num_node_features


def train_epoch(model, train_loader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(batch.x, batch.edge_index, batch.batch, return_attention=False)
        loss = criterion(logits, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, logger):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            
            logits, _ = model(batch.x, batch.edge_index, batch.batch, return_attention=False)
            loss = criterion(logits, batch.y)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def main(args):
    # Setup
    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("GAT Model Training")
    logger.info("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['reproducibility']['seed'])
    np.random.seed(config['reproducibility']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['reproducibility']['seed'])
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_features = create_data_loaders(config, logger)
    
    # Create model
    model = GAT(
        num_node_features=num_features,
        hidden_units=config['gat']['hidden_units'],
        num_heads=config['gat']['num_heads'],
        num_classes=2,
        dropout=config['gat']['dropout'],
        attention_dropout=config['gat']['attention_dropout']
    ).to(device)
    
    logger.info(f"Model created:")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Hidden units: {config['gat']['hidden_units']}")
    logger.info(f"  Num heads: {config['gat']['num_heads']}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['gat']['learning_rate'],
        weight_decay=config['gat']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Tensorboard
    writer = SummaryWriter(log_dir=Path(args.log_dir) / 'tensorboard')
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    
    for epoch in range(config['gat']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['gat']['epochs']}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, logger
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device, logger
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                   f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            model_path = Path(config['paths']['models']) / 'gat_best.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config['gat']
            }, model_path)
            
            logger.info(f"*** New best model saved! F1: {val_f1:.4f} ***")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['gat']['early_stopping_patience']:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model and evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Best Model on Test Set")
    logger.info("=" * 80)
    
    checkpoint = torch.load(Path(config['paths']['models']) / 'gat_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, test_loader, criterion, device, logger
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall: {test_rec:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    
    # Save final results
    results = {
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'config': config['gat']
    }
    
    import json
    results_path = Path(config['paths']['results']) / 'gat_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    writer.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT model for malware detection")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='logs/gat',
                       help='Directory for logs')
    
    args = parser.parse_args()
    main(args)