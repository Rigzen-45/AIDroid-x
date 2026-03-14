"""
Training Script for GAM Model - FIXED VERSION
Critical fixes:
1. Class-weighted loss
2. Proper hyperparameters (population=10, step_size=40)
3. Better evaluation
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.models.gam_model import GAMClassifier, GeneticAlgorithmTrainer
from src.dataset.graph_dataset import MalwareGraphDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train_gam.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_data_loaders(config, logger):
    """Create data loaders and compute class weights."""
    logger.info("Loading datasets...")
    
    # Load datasets
    train_graphs, train_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/train/graphs.pkl'
    )
    val_graphs, val_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/val/graphs.pkl'
    )
    test_graphs, test_labels = MalwareGraphDataset.load_dataset(
        config['paths']['graphs'] + '/test/graphs.pkl'
    )
    
    # CRITICAL: Compute class weights
    counter = Counter(train_labels)
    logger.info(f"Class distribution: {dict(counter)}")

    total = len(train_labels)
    class_weights = torch.FloatTensor([
        total / (2.0 * counter[0]),
        total / (2.0 * counter[1]),
    ])
    logger.info(f"Class weights: benign={class_weights[0]:.4f}, malware={class_weights[1]:.4f}")

    
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
    
    # Create loaders
    batch_size = config['gam'].get('batch_size', 32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}") 
    
    return train_loader, val_loader, test_loader, train_dataset.num_node_features, class_weights


def evaluate_model(model, data_loader, device, criterion=None, threshold=0.3):
    """Evaluate model with comprehensive metrics.
    
    Uses threshold=0.3 to match the operating point used during GAT training.
    Using argmax (≡0.5) here would report inflated Recall and deflated Precision
    compared to the model's actual calibrated operating point (docx Section 7.1).
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)

            probs = torch.softmax(logits, dim=1)
            malware_probs = probs[:, 1]
            preds = (malware_probs > threshold).long()
            
            if criterion is not None:
                loss = criterion(logits, batch.y)
                total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    avg_loss = total_loss / len(data_loader) if criterion else 0
    
    return accuracy, precision, recall, f1, cm, avg_loss


def main(args):
    logger = setup_logging(args.log_dir)
    logger.info("="*80)
    logger.info("GAM Model Training - FIXED")
    logger.info("="*80)
    
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set seeds
    torch.manual_seed(config['reproducibility']['seed'])
    np.random.seed(config['reproducibility']['seed'])
    
    # Load data with class weights
    train_loader, val_loader, test_loader, num_features, class_weights  = create_data_loaders(config, logger)
    
    # Initialize GA Trainer with class weights
    ga_trainer = GeneticAlgorithmTrainer(
        num_node_features=num_features,
        population_size=config['gam'].get('population_size', 20),
        step_size=config['gam'].get('step_size', 80),
        num_generations=config['gam'].get('num_generations', 30),
        mutation_rate=config['gam'].get('mutation_rate', 0.15),
        crossover_rate=config['gam'].get('crossover_rate', 0.7),
        tournament_size=config['gam'].get('tournament_size', 3),
        elitism=config['gam'].get('elitism', 2),
        device=device,
        class_weights=class_weights, 

    )
    
    logger.info(f"GA Configuration:")
    logger.info(f"  Population: {ga_trainer.population_size}")
    logger.info(f"  Step size: {ga_trainer.step_size}")
    logger.info(f"  Generations: {ga_trainer.num_generations}")
    logger.info(f"  Mutation rate: {ga_trainer.mutation_rate}")
    logger.info(f"  Crossover rate: {ga_trainer.crossover_rate}")
    
    writer = SummaryWriter(log_dir=Path(args.log_dir) / 'tensorboard')
    
    logger.info("\n" + "="*80)
    logger.info("Starting Evolutionary Training")
    logger.info("="*80)


    # [L-02 FIX] Pass writer into train() so TensorBoard scalars are written
    # live after every generation, not post-hoc after the full run completes.
    best_model = ga_trainer.train(train_loader, val_loader, writer=writer)

    logger.info("\nFitness history (all generations):")
    for generation, fitness in enumerate(ga_trainer.fitness_history):
        logger.info(
            f"  Generation {generation+1:>3}/{ga_trainer.num_generations} "
            f"Best Fitness: {fitness:.4f}"
        )

    # Save best model
    model_path = Path(config['paths']['models']) / 'gam_best.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_fitness':     ga_trainer.best_fitness,
        'fitness_history':  ga_trainer.fitness_history,
        'model_config': {
            'num_node_features': num_features,
            'hidden_dim':        64,
            'embedding_dim':     32,
            'num_classes':       2,
        },
        'config': config['gam'],
    }, model_path)
    
    logger.info(f"\nBest model saved: {model_path}")
    logger.info(f"Best fitness: {ga_trainer.best_fitness:.4f}")
    
    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("Testing Best Model")
    logger.info("="*80)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_acc, test_prec, test_rec, test_f1, test_cm, test_loss = evaluate_model(
        best_model, test_loader, device, criterion
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss:      {test_loss:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall:    {test_rec:.4f}")
    logger.info(f"  F1:        {test_f1:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  [[{test_cm[0][0]:4d} {test_cm[0][1]:4d}]")
    logger.info(f"   [{test_cm[1][0]:4d} {test_cm[1][1]:4d}]]")
    
    # Save results
    results = {
        'best_fitness': float(ga_trainer.best_fitness),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'test_confusion_matrix': test_cm.tolist(),
        'fitness_history': [float(f) for f in ga_trainer.fitness_history],
        'config': config['gam']
    }
    
    results_path = Path(config['paths']['results']) / 'gam_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved: {results_path}")
    writer.close()
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_dir', default='logs/gam')
    args = parser.parse_args()
    main(args)
