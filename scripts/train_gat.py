"""
GAT Training Script - COMPLETE FIX
Fixes applied:
- Class-weighted CrossEntropyLoss (audit fix)
- Gradient norm logging before clip (M-05)
- ReduceLROnPlateau replacing cosine schedule (H-06)
- Linear warmup for first WARMUP_EPOCHS before scheduler takes over
  (prevents BatchNorm instability on cold AdamW start with small datasets)
- Reduced default hidden_units/num_heads for small-dataset regime
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from collections import Counter
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.gat_model import GAT
from src.dataset.graph_dataset import MalwareGraphDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Number of epochs to linearly ramp LR from 0 → base_lr before
# handing control to ReduceLROnPlateau.
# Prevents corrupted weight initialisation from noisy first-batch gradients
# when BatchNorm has not yet accumulated stable statistics (critical on
# small datasets where each batch has high variance).
WARMUP_EPOCHS = 5   # fallback if config missing warmup_epochs key


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_data_loaders(config, logger):
    logger.info("Loading datasets...")

    train_graphs, train_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/train/graphs.pkl'
    )
    val_graphs, val_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/val/graphs.pkl'
    )
    test_graphs, test_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/test/graphs.pkl'
    )

    train_counter = Counter(train_labels)
    logger.info(f"Class distribution: {dict(train_counter)}")

    total = len(train_labels)
    class_weights = torch.FloatTensor([
        total / (2.0 * train_counter[0]),
        total / (2.0 * train_counter[1]),
    ])
    logger.info(f"Class weights: benign={class_weights[0]:.4f}, malware={class_weights[1]:.4f}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['gat'].get('batch_size', 32),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['gat'].get('batch_size', 32),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['gat'].get('batch_size', 32),
        shuffle=False,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── Feature sanity check ─────────────────────────────────────────────────
    # Verifies that graph_builder populated category_counts correctly.
    # If Group1 non-zero nodes = 0, the processed/ cache is stale:
    #   delete data/graphs/*/processed/ and re-run preprocess_apks.py.
    _b = next(iter(train_loader))
    logger.info(f"[SANITY] Feature shape          : {list(_b.x.shape)}  (expected [N, 41])")
    logger.info(f"[SANITY] Group1 non-zero nodes  : {(_b.x[:, :10].sum(1) > 0).sum().item()} / {_b.x.shape[0]}"
                f"  (must be > 0 — if 0, delete processed/ cache)")
    logger.info(f"[SANITY] Feature means dim 0-9  : {[round(v, 4) for v in _b.x.mean(0)[:10].tolist()]}"
                f"  (must NOT all be 0.0)")
    logger.info(f"[SANITY] Batch class distribution: {_b.y.bincount().tolist()}")
    # ── End sanity check ─────────────────────────────────────────────────────

    return train_loader, val_loader, test_loader, train_dataset.num_node_features, class_weights


def train_epoch(model, loader, optimizer, criterion, device, epoch, writer, global_step):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Pass graph_size if present (added by graph_dataset.py to address
        # the 7.57× malware/benign size differential)
        graph_size = getattr(batch, "graph_size", None)

        logits, _ = model(
            batch.x, batch.edge_index, batch.batch,
            return_attention=False,
            graph_size=graph_size,
        )
        loss = criterion(logits, batch.y)
        loss.backward()

        # [M-05] Log gradient norm BEFORE clipping for training diagnostics.
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradients/norm_before_clip', total_norm, global_step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
        global_step += 1

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), global_step


def validate(model, loader, criterion, device, threshold=0.3):
    """Validate model. threshold=0.3 matches the training operating point.
    
    Using 0.3 here is critical — the model is early-stopped on val_f1 computed
    at this threshold, so evaluate.py must also use 0.3 for consistent metrics
    (docx Section 5.1 and 7.1).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            graph_size = getattr(batch, "graph_size", None)

            logits, _ = model(
                batch.x, batch.edge_index, batch.batch,
                return_attention=False,
                graph_size=graph_size,
            )
            loss = criterion(logits, batch.y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            malware_probs = probs[:, 1]
            preds = (malware_probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(malware_probs.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, accuracy, precision, recall, f1, auc, cm


def main(args):
    logger = setup_logging(args.log_dir)

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {config['gat']['learning_rate']}")
    logger.info(f"Warmup epochs: {WARMUP_EPOCHS}")

    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, num_features, class_weights = create_data_loaders(config, logger)

    model = GAT(
        num_node_features=num_features,
        hidden_units=config['gat'].get('hidden_units', 32),
        num_heads=config['gat'].get('num_heads', 8),
        num_classes=2,
        dropout=config['gat'].get('dropout', 0.3),
        attention_dropout=config['gat'].get('attention_dropout', 0.4),
        use_graph_size=True,   # concat log1p(nodes,edges) before FC head
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['gat']['learning_rate'],
        weight_decay=config['gat']['weight_decay'],
    )

    # ReduceLROnPlateau takes over after warmup completes.
    # patience=5: tolerates 5 stagnant epochs before halving LR.
    # mode='max': tracks val_f1 (higher is better).
    # min_lr=1e-5: floor prevents LR from dropping so low the model
    # stops learning before convergence (docx Section 5.2).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=True,
    )

    writer = SummaryWriter(log_dir=Path(args.log_dir) / 'tensorboard')

    best_val_f1      = 0.0
    best_epoch       = 0
    patience_counter = 0
    global_step      = 0
    total_epochs     = config['gat']['epochs']
    base_lr          = config['gat']['learning_rate']

    # Read warmup length from config so it stays in sync with config.yaml
    # (docx Section 10: warmup_epochs 5→8 for 10K dataset).
    warmup_epochs = config['gat'].get('warmup_epochs', WARMUP_EPOCHS)

    for epoch in range(total_epochs):

        # ── Learning rate: warmup → plateau scheduler ─────────────────────
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']
        # ── End LR logic ──────────────────────────────────────────────────

        logger.info(f"\nEpoch {epoch+1}/{total_epochs} (LR: {current_lr:.6f})")
        writer.add_scalar('LR', current_lr, epoch)

        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1, writer, global_step
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion, device, threshold=0.4
        )


        logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
                    f"F1={val_f1:.4f}, AUC={val_auc:.4f}, "
                    f"Prec={val_prec:.4f}, Rec={val_rec:.4f}")

        if epoch >= warmup_epochs:
            scheduler.step(val_f1)

        # TensorBoard logging
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val',         val_loss,   epoch)
        writer.add_scalar('Metrics/val_acc',       val_acc,  epoch)
        writer.add_scalar('Metrics/val_f1',        val_f1,   epoch)
        writer.add_scalar('Metrics/val_auc',       val_auc,  epoch)
        writer.add_scalar('Metrics/val_precision', val_prec, epoch)
        writer.add_scalar('Metrics/val_recall',    val_rec,  epoch)

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_epoch       = epoch
            patience_counter = 0

            model_path = Path(config['paths']['models']) / 'gat_best.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'model_config': {
                    'num_node_features': num_features,
                    'hidden_units':      config['gat'].get('hidden_units', 32),
                    'num_heads':         config['gat'].get('num_heads', 4),
                    'num_classes':       2,
                    'dropout':           config['gat'].get('dropout', 0.3),
                    'attention_dropout': config['gat'].get('attention_dropout', 0.3),
                },
            }, model_path)
            logger.info(f"Best model saved (Epoch {epoch+1}, F1={val_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config['gat'].get('early_stopping_patience', 25):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # ── Test ─────────────────────────────────────────────────────────────────
    logger.info("\nTesting best model")
    checkpoint = torch.load(Path(config['paths']['models']) / 'gat_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_cm = validate(
        model, test_loader, criterion, device
    )

    logger.info(f"Test Accuracy:  {test_acc:.4f}")
    logger.info(f"Test Precision: {test_prec:.4f}")
    logger.info(f"Test Recall:    {test_rec:.4f}")
    logger.info(f"Test F1:        {test_f1:.4f}")
    logger.info(f"Test AUC:       {test_auc:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_cm}")

    results = {
        'best_epoch':            int(checkpoint['epoch'] + 1),
        'test_accuracy':         float(test_acc),
        'test_precision':        float(test_prec),
        'test_recall':           float(test_rec),
        'test_f1':               float(test_f1),
        'test_auc':              float(test_auc),
        'test_confusion_matrix': test_cm.tolist(),
    }

    results_path = Path(config['paths']['results']) / 'gat_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_path}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs/gat')
    args = parser.parse_args()
    main(args)
