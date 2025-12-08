"""
Training Script for GAM Model (Genetic Algorithm Module)
Paper params: population=10, step_size=40, generations=50
"""

import argparse
import yaml
import logging
import os
import sys
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gam_model import GAMClassifier, GeneticAlgorithmTrainer
from src.dataset.graph_dataset import MalwareGraphDataset


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train_gam.log'),
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
        batch_size=config['gam'].get('batch_size', 32),
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['gam'].get('batch_size', 32),
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['gam'].get('batch_size', 32),
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val: {len(val_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, train_dataset.num_node_features


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return accuracy, precision, recall, f1


def main(args):
    # Setup
    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("GAM Model Training (Genetic Algorithm)")
    logger.info("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config['reproducibility']['seed'])
    np.random.seed(config['reproducibility']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['reproducibility']['seed'])
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_features = create_data_loaders(config, logger)
    
    # Initialize Genetic Algorithm Trainer
    ga_trainer = GeneticAlgorithmTrainer(
        num_node_features=num_features,
        population_size=config['gam']['population_size'],
        step_size=config['gam']['step_size'],
        num_generations=config['gam']['num_generations'],
        mutation_rate=config['gam']['mutation_rate'],
        crossover_rate=config['gam']['crossover_rate'],
        tournament_size=config['gam']['tournament_size'],
        elitism=config['gam']['elitism'],
        device=device
    )
    
    logger.info(f"Genetic Algorithm Configuration:")
    logger.info(f"  Population size: {config['gam']['population_size']}")
    logger.info(f"  Step size: {config['gam']['step_size']}")
    logger.info(f"  Generations: {config['gam']['num_generations']}")
    logger.info(f"  Mutation rate: {config['gam']['mutation_rate']}")
    logger.info(f"  Crossover rate: {config['gam']['crossover_rate']}")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=Path(args.log_dir) / 'tensorboard')
    
    # Train using genetic algorithm
    logger.info("\n" + "=" * 80)
    logger.info("Starting Evolutionary Training")
    logger.info("=" * 80)
    
    best_model = ga_trainer.train(train_loader, val_loader)
    
    # Log fitness history
    for gen, fitness in enumerate(ga_trainer.fitness_history):
        writer.add_scalar('Fitness/best', fitness, gen)
    
    # Save best model
    model_path = Path(config['paths']['models']) / 'gam_best.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_fitness': ga_trainer.best_fitness,
        'fitness_history': ga_trainer.fitness_history,
        'config': config['gam']
    }, model_path)
    
    logger.info(f"\nBest model saved to {model_path}")
    logger.info(f"Best fitness: {ga_trainer.best_fitness:.4f}")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Best Model on Test Set")
    logger.info("=" * 80)
    
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(
        best_model, test_loader, device
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall: {test_rec:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    
    # Save results
    results = {
        'best_fitness': float(ga_trainer.best_fitness),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'fitness_history': [float(f) for f in ga_trainer.fitness_history],
        'config': config['gam']
    }
    
    results_path = Path(config['paths']['results']) / 'gam_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    writer.close()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAM model using genetic algorithm")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='logs/gam',
                       help='Directory for logs')
    
    args = parser.parse_args()
    main(args)