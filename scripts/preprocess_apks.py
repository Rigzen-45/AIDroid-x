"""
APK Preprocessing Pipeline
Complete workflow: APK → Decompile → Graph → PyG Dataset
"""

import argparse
import logging
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import json
import networkx as nx

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.apk_decompiler import APKDecompiler
from src.preprocessing.api_extractor import SensitiveAPIFilter
from src.preprocessing.graph_builder import APICallGraphBuilder
from src.dataset.graph_dataset import MalwareGraphDataset

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def process_single_apk(apk_path: str, label: int, 
                       api_filter: SensitiveAPIFilter,
                       graph_builder: APICallGraphBuilder,
                       intermediate_dir: Path) -> Tuple:
    apk_name = Path(apk_path).stem
    """
    Process a single APK file.
    
    Args:
        apk_path: Path to APK file
        label: Label (0=benign, 1=malware)
        api_filter: SensitiveAPIFilter instance
        graph_builder: APICallGraphBuilder instance
        
    Returns:
        Tuple of (graph, label, success, apk_name)
    """
    try:
        # Decompile APK
        decompiler = APKDecompiler(output_dir=intermediate_dir/"decompiled")
        decompiled = decompiler.decompile(apk_path)
        
        if not decompiled["success"]:
            return None, None, False, apk_name
        
        # Save raw API calls
        api_out = intermediate_dir / "api_calls"
        api_out.mkdir(parents=True, exist_ok=True)

        with open(api_out / f"{apk_name}.json", "w") as f:
            json.dump(decompiled["api_calls"], f, indent=2)


        # Filter to sensitive APIs only
        filtered = api_filter.filter_api_calls(decompiled["api_calls"])
        
        if filtered["total_sensitive"] < 1:
            logging.warning(f"Skipping {apk_name}: "
                          f"Only {filtered['total_sensitive']} sensitive APIs")
            return None, None, False, apk_name
        
        # Build API call graph
        decompiled["label"] = "malware" if label == 1 else "benign"
        graph = graph_builder.build_graph(decompiled)
        
        # Check graph validity
        if graph.number_of_nodes() < 10:
            logging.warning(f"Skipping {apk_name}: "
                          f"Graph too small ({graph.number_of_nodes()} nodes)")
            return None, None, False, apk_name
        
        # Save raw graph visually 
        graph_dir = intermediate_dir / "graphs_raw"
        graph_dir.mkdir(parents=True, exist_ok=True)

        nx.write_gexf(graph, graph_dir / f"{apk_name}.gexf")

        return graph, label, True, apk_name
        
    except Exception as e:
        logging.error(f"Error processing {apk_name}: {e}")
        return None, None, False, apk_name


def process_apk_directory(apk_dir: str, label: int, 
                         api_filter: SensitiveAPIFilter,
                         graph_builder: APICallGraphBuilder,
                         n_workers: int = 4) -> Tuple[List, List]:
    """
    Process all APKs in a directory with parallel processing.
    
    Args:
        apk_dir: Directory containing APK files
        label: Label for all APKs (0=benign, 1=malware)
        api_filter: SensitiveAPIFilter instance
        graph_builder: APICallGraphBuilder instance
        n_workers: Number of parallel workers
        
    Returns:
        Tuple of (graphs, labels)
    """
    apk_dir = Path(apk_dir)
    apk_files = list(apk_dir.glob("*.apk"))
    
    logging.info(f"Processing {len(apk_files)} APKs from {apk_dir}")
    
    graphs = []
    labels = []
    
    # Note: ProcessPoolExecutor doesn't work well with complex objects
    # Process sequentially with progress bar instead
    for apk_path in tqdm(apk_files, desc=f"Processing {apk_dir.name}"):
        graph, lbl, success, apk_name = process_single_apk(
            str(apk_path), label, api_filter, graph_builder,
            Path("data/intermediate")
        )
        
        if success:
            graphs.append(graph)
            labels.append(lbl)
    
    logging.info(f"Successfully processed {len(graphs)}/{len(apk_files)} APKs")
    
    return graphs, labels


def split_dataset(graphs: List, labels: List, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42) -> Tuple:
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        graphs: List of graphs
        labels: List of labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
        
    Returns:
        Tuple of (train_graphs, train_labels, val_graphs, val_labels, 
                 test_graphs, test_labels)
    """
    np.random.seed(random_seed)
    
    # Separate by class for stratification
    malware_indices = [i for i, l in enumerate(labels) if l == 1]
    benign_indices = [i for i, l in enumerate(labels) if l == 0]
    
    np.random.shuffle(malware_indices)
    np.random.shuffle(benign_indices)
    
    # Calculate split points
    def split_indices(indices, train_r, val_r):
        n = len(indices)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    malware_train, malware_val, malware_test = split_indices(
        malware_indices, train_ratio, val_ratio
    )
    benign_train, benign_val, benign_test = split_indices(
        benign_indices, train_ratio, val_ratio
    )
    
    # Combine and shuffle
    train_idx = malware_train + benign_train
    val_idx = malware_val + benign_val
    test_idx = malware_test + benign_test
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Extract data
    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_graphs = [graphs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    logging.info(f"Dataset split:")
    logging.info(f"  Train: {len(train_graphs)} samples "
                f"(malware: {sum(train_labels)}, benign: {len(train_labels) - sum(train_labels)})")
    logging.info(f"  Val: {len(val_graphs)} samples "
                f"(malware: {sum(val_labels)}, benign: {len(val_labels) - sum(val_labels)})")
    logging.info(f"  Test: {len(test_graphs)} samples "
                f"(malware: {sum(test_labels)}, benign: {len(test_labels) - sum(test_labels)})")
    
    return (train_graphs, train_labels, val_graphs, val_labels, 
            test_graphs, test_labels)


def main(args):
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("APK Preprocessing Pipeline")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("Initializing preprocessing components...")
    api_filter = SensitiveAPIFilter(args.sensitive_apis)
    graph_builder = APICallGraphBuilder(api_filter)
    
    # Process malware APKs
    logger.info("\n" + "=" * 80)
    logger.info("Processing Malware APKs")
    logger.info("=" * 80)
    malware_graphs, malware_labels = process_apk_directory(
        args.malware_dir, label=1, 
        api_filter=api_filter,
        graph_builder=graph_builder,
        n_workers=args.n_workers
    )
    
    # Process benign APKs
    logger.info("\n" + "=" * 80)
    logger.info("Processing Benign APKs")
    logger.info("=" * 80)
    benign_graphs, benign_labels = process_apk_directory(
        args.benign_dir, label=0,
        api_filter=api_filter,
        graph_builder=graph_builder,
        n_workers=args.n_workers
    )
    
    # Combine datasets
    all_graphs = malware_graphs + benign_graphs
    all_labels = malware_labels + benign_labels
    
    logger.info(f"\nTotal processed: {len(all_graphs)} APKs")
    logger.info(f"  Malware: {sum(all_labels)}")
    logger.info(f"  Benign: {len(all_labels) - sum(all_labels)}")
    
    # Split dataset
    logger.info("\n" + "=" * 80)
    logger.info("Splitting Dataset")
    logger.info("=" * 80)
    
    (train_graphs, train_labels, val_graphs, val_labels, 
     test_graphs, test_labels) = split_dataset(
        all_graphs, all_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Save processed data
    logger.info("\n" + "=" * 80)
    logger.info("Saving Processed Datasets")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    
    # Save train set
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(
        train_graphs, train_labels, 
        str(train_dir / 'graphs.pkl')
    )
    
    # Save val set
    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(
        val_graphs, val_labels,
        str(val_dir / 'graphs.pkl')
    )
    
    # Save test set
    test_dir = output_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(
        test_graphs, test_labels,
        str(test_dir / 'graphs.pkl')
    )
    
    logger.info(f"Datasets saved to {output_dir}")
    
    # Save preprocessing statistics
    stats = {
        'total_apks': len(all_graphs),
        'malware_count': sum(all_labels),
        'benign_count': len(all_labels) - sum(all_labels),
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'avg_nodes': np.mean([g.number_of_nodes() for g in all_graphs]),
        'avg_edges': np.mean([g.number_of_edges() for g in all_graphs]),
    }
    
    import json
    stats_path = output_dir / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nStatistics saved to {stats_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Train GAT: python scripts/train_gat.py")
    logger.info(f"2. Train GAM: python scripts/train_gam.py")
    logger.info(f"3. Run inference: python scripts/inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess APK files for XAIDroid")
    parser.add_argument('--malware_dir', type=str, required=True,
                       help='Directory containing malware APKs')
    parser.add_argument('--benign_dir', type=str, required=True,
                       help='Directory containing benign APKs')
    parser.add_argument('--output_dir', type=str, default='data/graphs',
                       help='Output directory for processed graphs')
    parser.add_argument('--sensitive_apis', type=str, 
                       default='config/sensitive_apis.json',
                       help='Path to sensitive APIs JSON')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Train/val/test ratios must sum to 1.0")
    
    main(args)
