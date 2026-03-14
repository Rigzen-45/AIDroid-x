"""
APK Preprocessing Pipeline
Complete workflow: APK → Decompile → Graph → PyG Dataset
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import json
import networkx as nx

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.apk_decompiler import APKDecompiler
from src.preprocessing.api_extractor import SensitiveAPIFilter
from src.preprocessing.graph_builder import APICallGraphBuilder
from src.dataset.graph_dataset import MalwareGraphDataset


def setup_logging(log_dir: str):
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
                       decompiler: APKDecompiler,          # FIX 2: passed in, not created here
                       intermediate_dir: Path) -> Tuple:
    """
    Process a single APK file.
    Returns: Tuple of (graph, label, success, apk_name)
    """
    apk_name = Path(apk_path).stem

    try:
        # Decompile APK
        decompiled = decompiler.decompile(apk_path)

        if not decompiled["success"]:
            return None, None, False, apk_name

        # FIX 4: Save filtered api_calls (not raw) to avoid ~5GB disk usage.
        # Raw calls can be 6000+ per APK; filtered is typically <700.
        filtered = api_filter.filter_api_calls(decompiled)

        if filtered["total_sensitive"] < 1:
            logging.warning(f"Skipping {apk_name}: "
                            f"Only {filtered['total_sensitive']} sensitive APIs")
            return None, None, False, apk_name

        # Save filtered API calls for debugging/inspection
        api_out = intermediate_dir / "api_calls"
        api_out.mkdir(parents=True, exist_ok=True)
        with open(api_out / f"{apk_name}.json", "w") as f:
            json.dump(filtered["methods"], f, indent=2)

        # Build graph from filtered APIs
        decompiled["api_calls"] = filtered["methods"]
        decompiled["label"] = "malware" if label == 1 else "benign"
        graph = graph_builder.build_graph(decompiled)

        # Check graph validity
        if graph.number_of_nodes() < 3:
            logging.warning(f"Skipping {apk_name}: "
                            f"Graph too small ({graph.number_of_nodes()} nodes)")
            return None, None, False, apk_name

        # FIX 6: Write GEXF from a COPY so the live graph object is not mutated.
        # Mutating graph.nodes[n]["viz"] in-place contaminates the object that
        # goes into graphs[] -> graphs.pkl -> graph_dataset.py.
        graph_dir = intermediate_dir / "graphs_raw"
        graph_dir.mkdir(parents=True, exist_ok=True)

        gexf_graph = graph.copy()
        for node, data in gexf_graph.nodes(data=True):
            is_sens = data.get('is_sensitive', False) or data.get('has_sensitive_api', False)
            data['viz'] = {
                'color': {'r': 255, 'g': 0, 'b': 0} if is_sens else {'r': 180, 'g': 180, 'b': 180},
                'size': gexf_graph.degree(node) * 3 + 10
            }
            if '->' in str(node):
                parts = str(node).split('->')
                class_short = parts[0].split('/')[-1].replace(';', '')
                method_short = parts[1].split('(')[0] if len(parts) > 1 else ''
                data['label'] = f"{class_short}.{method_short}"
            else:
                data['label'] = str(node)[-40:]

        nx.write_gexf(gexf_graph, graph_dir / f"{apk_name}.gexf")

        return graph, label, True, apk_name

    except Exception as e:
        logging.error(f"Error processing {apk_name}: {e}")
        return None, None, False, apk_name


def process_apk_directory(
    apk_dir: str,
    label: int,
    api_filter: SensitiveAPIFilter,
    graph_builder: APICallGraphBuilder,
    output_dir: str
) -> Tuple[List, List]:
    """Process all APKs in a directory sequentially."""

    apk_dir = Path(apk_dir)
    intermediate_dir = Path(output_dir) / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # FIX 2: Create APKDecompiler ONCE per directory, not once per APK
    decompiler = APKDecompiler(output_dir=intermediate_dir / "decompiled")

    apk_files = list(apk_dir.glob("*.apk"))
    logging.info(f"Processing {len(apk_files)} APKs from {apk_dir}")

    from tqdm import tqdm
    graphs = []
    labels = []

    for apk_path in tqdm(apk_files, desc=f"Processing {apk_dir.name}"):
        graph, lbl, success, apk_name = process_single_apk(
            str(apk_path),
            label,
            api_filter,
            graph_builder,
            decompiler,           # FIX 2: pass shared decompiler
            intermediate_dir
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
    """Split dataset into train/val/test with stratification."""
    np.random.seed(random_seed)

    malware_indices = [i for i, l in enumerate(labels) if l == 1]
    benign_indices  = [i for i, l in enumerate(labels) if l == 0]

    np.random.shuffle(malware_indices)
    np.random.shuffle(benign_indices)

    def split_indices(indices, train_r, val_r):
        n = len(indices)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    malware_train, malware_val, malware_test = split_indices(malware_indices, train_ratio, val_ratio)
    benign_train,  benign_val,  benign_test  = split_indices(benign_indices,  train_ratio, val_ratio)

    train_idx = malware_train + benign_train
    val_idx   = malware_val   + benign_val
    test_idx  = malware_test  + benign_test

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    logging.info("Dataset split:")
    logging.info(f"  Train: {len(train_graphs)} "
                 f"(malware: {sum(train_labels)}, benign: {len(train_labels)-sum(train_labels)})")
    logging.info(f"  Val:   {len(val_graphs)} "
                 f"(malware: {sum(val_labels)}, benign: {len(val_labels)-sum(val_labels)})")
    logging.info(f"  Test:  {len(test_graphs)} "
                 f"(malware: {sum(test_labels)}, benign: {len(test_labels)-sum(test_labels)})")

    return (train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels)


def main(args):
    logger = setup_logging(args.log_dir)
    logger.info("=" * 80)
    logger.info("APK Preprocessing Pipeline")
    logger.info("=" * 80)

    # FIX 3: Pass output_dir-aware paths to APICallGraphBuilder
    intermediate_dir = Path(args.output_dir) / "intermediate"
    reports_dir = intermediate_dir / "reports"
    viz_dir     = intermediate_dir / "viz"

    logger.info("Initializing preprocessing components...")
    api_filter    = SensitiveAPIFilter(args.sensitive_apis)
    graph_builder = APICallGraphBuilder(
        api_filter,
        reports_dir=str(reports_dir),   # FIX 3: respects --output_dir
        viz_dir=str(viz_dir),
    )

    # Process malware APKs
    logger.info("\n" + "=" * 80)
    logger.info("Processing Malware APKs")
    logger.info("=" * 80)
    malware_graphs, malware_labels = process_apk_directory(
        args.malware_dir, label=1,
        api_filter=api_filter,
        graph_builder=graph_builder,
        output_dir=args.output_dir,
    )

    # Process benign APKs
    logger.info("\n" + "=" * 80)
    logger.info("Processing Benign APKs")
    logger.info("=" * 80)
    benign_graphs, benign_labels = process_apk_directory(
        args.benign_dir, label=0,
        api_filter=api_filter,
        graph_builder=graph_builder,
        output_dir=args.output_dir,
    )

    # Combine
    all_graphs = malware_graphs + benign_graphs
    all_labels = malware_labels + benign_labels

    logger.info(f"\nTotal processed: {len(all_graphs)} APKs")
    logger.info(f"  Malware: {sum(all_labels)}")
    logger.info(f"  Benign:  {len(all_labels) - sum(all_labels)}")

    # Split
    logger.info("\n" + "=" * 80)
    logger.info("Splitting Dataset")
    logger.info("=" * 80)
    (train_graphs, train_labels,
     val_graphs,   val_labels,
     test_graphs,  test_labels) = split_dataset(
        all_graphs, all_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Save
    logger.info("\n" + "=" * 80)
    logger.info("Saving Processed Datasets")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)

    for split_name, split_graphs, split_labels in [
        ('train', train_graphs, train_labels),
        ('val',   val_graphs,   val_labels),
        ('test',  test_graphs,  test_labels),
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        MalwareGraphDataset.save_dataset(
            split_graphs, split_labels,
            str(split_dir / 'graphs.pkl')
        )

    logger.info(f"Datasets saved to {output_dir}")

    stats = {
        'total_apks':    len(all_graphs),
        'malware_count': sum(all_labels),
        'benign_count':  len(all_labels) - sum(all_labels),
        'train_size':    len(train_graphs),
        'val_size':      len(val_graphs),
        'test_size':     len(test_graphs),
        'avg_nodes':     float(np.mean([g.number_of_nodes() for g in all_graphs])),
        'avg_edges':     float(np.mean([g.number_of_edges() for g in all_graphs])),
    }
    stats_path = output_dir / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_path}")
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Train GAT: python scripts/train_gat.py")
    logger.info("2. Train GAM: python scripts/train_gam.py")
    logger.info("3. Run inference: python scripts/inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess APK files for XAIDroid")
    parser.add_argument('--malware_dir',    type=str, required=True)
    parser.add_argument('--benign_dir',     type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default='data/graphs')
    parser.add_argument('--sensitive_apis', type=str, default='config/sensitive_apis.json')
    parser.add_argument('--train_ratio',    type=float, default=0.7)
    parser.add_argument('--val_ratio',      type=float, default=0.15)
    parser.add_argument('--test_ratio',     type=float, default=0.15)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--log_dir',        type=str,   default='logs')
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    main(args)
