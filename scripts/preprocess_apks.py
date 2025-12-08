#!/usr/bin/env python
"""
Preprocessing pipeline for XAIDroid
Updated to integrate upgraded APKDecompiler + SensitiveAPIFilter,
save per-APK reports, debug logs, and visualization of API category distribution.
"""
import argparse
import logging
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle
from tqdm import tqdm
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.apk_decompiler import APKDecompiler
from src.preprocessing.api_extractor import SensitiveAPIFilter
from src.preprocessing.graph_builder import APICallGraphBuilder
from src.dataset.graph_dataset import MalwareGraphDataset

# Optional visualization (matplotlib). If not installed, visuals will be skipped.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def setup_logging(log_dir: str, level=logging.INFO):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def is_probable_apk(path: Path) -> bool:
    """Quick magic-bytes check for ZIP/APK files."""
    try:
        with open(path, 'rb') as f:
            sig = f.read(4)
            return sig == b'PK\x03\x04'
    except Exception:
        return False


def extract_base_apk_if_bundle(path: Path) -> Path:
    """
    Some downloads (APKMirror) wrap multiple split apks into a single archive or apk-like file.
    This function tries to detect and extract a 'base.apk' (or the largest .apk inside)
    into a temporary file and returns its Path. If not a bundle, returns the original path.
    """
    try:
        if not zipfile.is_zipfile(path):
            return path
    except Exception:
        return path

    tmpdir = Path(tempfile.mkdtemp(prefix='xaidroid_apk_'))
    try:
        with zipfile.ZipFile(path, 'r') as z:
            members = z.namelist()
            candidates = [m for m in members if m.lower().endswith('.apk')]
            base_candidate = None
            if 'base.apk' in members:
                base_candidate = 'base.apk'
            elif len(candidates) == 1:
                base_candidate = candidates[0]
            elif len(candidates) > 1:
                largest = None
                largest_size = -1
                for m in candidates:
                    info = z.getinfo(m)
                    if info.file_size > largest_size:
                        largest = m
                        largest_size = info.file_size
                base_candidate = largest

            if base_candidate:
                extracted = tmpdir / 'extracted_base.apk'
                with z.open(base_candidate) as src, open(extracted, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                return extracted
    except Exception:
        # If extraction failed, return original path
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        return path

    # cleanup if nothing returned
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
    return path


def visualize_api_distribution(dist: Dict[str, float], save_path: Path, title: str = "API Category Distribution"):
    """Save a barplot showing category percentages. Requires matplotlib."""
    if not _HAS_MPL:
        logging.getLogger(__name__).warning("matplotlib not installed; skipping visualization.")
        return
    try:
        categories = list(dist.keys())
        values = [dist[c] for c in categories]
        plt.figure(figsize=(10, 5))
        plt.bar(categories, values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Percentage (%)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logging.getLogger(__name__).debug(f"Visualization failed: {e}", exc_info=True)


def safe_process_single_apk(apk_path: str, label: int,
                            api_filter: SensitiveAPIFilter,
                            graph_builder: APICallGraphBuilder,
                            reports_dir: Path) -> Tuple:
    """
    Wrapper that attempts preprocessing on a single APK with bundle handling.
    Returns (graph, label, success, apk_name, reason)
    """
    apk_path = Path(apk_path)
    apk_name = apk_path.name

    logger = logging.getLogger(__name__)

    # Quick magic check
    if not is_probable_apk(apk_path):
        logger.warning(f"Skipping {apk_name}: not a valid ZIP/APK (magic bytes)")
        return None, None, False, apk_name, 'invalid_magic'

    # If bundle-like, extract base apk
    processed_path = extract_base_apk_if_bundle(apk_path)
    temp_extracted_dir = None
    if processed_path != apk_path:
        logger.info(f"Extracted base APK for {apk_name} -> {processed_path.name}")
        temp_extracted_dir = processed_path.parent

    try:
        decompiler = APKDecompiler()
        decompiled = decompiler.decompile(str(processed_path))
    except Exception as e:
        logger.error(f"Decompiler crashed for {apk_name}: {e}", exc_info=True)
        # cleanup extracted
        if temp_extracted_dir:
            shutil.rmtree(temp_extracted_dir, ignore_errors=True)
        return None, None, False, apk_name, 'decompiler_crash'

    # cleanup extracted temp files if any
    if temp_extracted_dir:
        try:
            shutil.rmtree(temp_extracted_dir, ignore_errors=True)
        except Exception:
            pass

    # If decompiler failed
    if not decompiled.get('success', False):
        logger.warning(f"Skipping {apk_name}: {decompiled.get('error')}")
        # save a short report to help debugging
        try:
            rpt_path = reports_dir / f"{apk_name}.json"
            reports_dir.mkdir(parents=True, exist_ok=True)
            with open(rpt_path, 'w') as rf:
                json.dump({"apk_name": apk_name, "error": decompiled.get('error'), "embedded_files": decompiled.get('embedded_files')}, rf, indent=2)
        except Exception:
            pass
        return None, None, False, apk_name, decompiled.get('error')

    # Run the extractor on the full decompiler result (so reflection evidence is used)
    filtered = api_filter.filter_api_calls(decompiled)

    # Save per-apk report (decompiler + filtered)
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        rpt = {
            "apk_name": apk_name,
            "package_name": decompiled.get("package_name"),
            "manifest_ok": decompiled.get("manifest_ok", False),
            "embedded_files": decompiled.get("embedded_files", []),
            "native_libs": decompiled.get("native_libs", []),
            "total_methods": len(decompiled.get("methods", [])),
            "total_sensitive": filtered.get("total_sensitive", 0),
            "category_stats": filtered.get("category_stats", {}),
            "reflection_evidence": decompiled.get("reflection_evidence", {}),
            "filtered_api_sample": {k: filtered['api_stats'][k] for k in list(filtered.get("api_stats", {}).keys())[:10]}
        }
        with open(reports_dir / f"{apk_name}.json", 'w') as rf:
            json.dump(rpt, rf, indent=2)
        # visualization per-apk (optional)
        try:
            viz_path = reports_dir / f"{apk_name}_category_dist.png"
            visualize_api_distribution(filtered.get("category_stats", {}), viz_path, title=f"{apk_name} category dist")
        except Exception:
            pass
    except Exception:
        logger.debug("Failed to write per-apk report", exc_info=True)

    # Heuristics: skip apps with too few sensitive APIs
    if filtered.get('total_sensitive', 0) < 5:
        logger.warning(f"Skipping {apk_name}: Only {filtered.get('total_sensitive',0)} sensitive APIs")
        return None, None, False, apk_name, 'few_sensitive_apis'

    # Build graph (graph_builder expects decompiler result)
    # graph_builder should use filtered info via api_filter if needed
    decompiled['label'] = 'malware' if label == 1 else 'benign'
    try:
        graph = graph_builder.build_graph(decompiled)
    except Exception as e:
        logger.error(f"Graph builder failed for {apk_name}: {e}", exc_info=True)
        return None, None, False, apk_name, 'graph_build_failed'

    if graph.number_of_nodes() < 10:
        logger.warning(f"Skipping {apk_name}: Graph too small ({graph.number_of_nodes()} nodes)")
        return None, None, False, apk_name, 'graph_too_small'

    return graph, label, True, apk_name, 'ok'


def process_apk_directory(apk_dir: str, label: int,
                          api_filter: SensitiveAPIFilter,
                          graph_builder: APICallGraphBuilder,
                          n_workers: int = 4,
                          reports_dir: Path = Path("data/reports")) -> Tuple[List, List]:
    apk_dir = Path(apk_dir)
    apk_files = list(apk_dir.glob('*.apk'))

    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(apk_files)} APKs from {apk_dir}")

    graphs = []
    labels = []

    # Sequential processing with progress bar (more stable for debugging)
    for apk_path in tqdm(apk_files, desc=f"Processing {apk_dir.name}"):
        graph, lbl, success, apk_name, reason = safe_process_single_apk(
            str(apk_path), label, api_filter, graph_builder, reports_dir
        )

        if success:
            graphs.append(graph)
            labels.append(lbl)
        else:
            logger.debug(f"Skipped {apk_name}: {reason}")

    logger.info(f"Successfully processed {len(graphs)}/{len(apk_files)} APKs")
    return graphs, labels


def split_dataset(graphs: List, labels: List,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple:
    np.random.seed(random_seed)

    if len(graphs) == 0:
        return [], [], [], [], [], []

    malware_indices = [i for i, l in enumerate(labels) if l == 1]
    benign_indices = [i for i, l in enumerate(labels) if l == 0]

    np.random.shuffle(malware_indices)
    np.random.shuffle(benign_indices)

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

    train_idx = malware_train + benign_train
    val_idx = malware_val + benign_val
    test_idx = malware_test + benign_test

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    def pick(idxs):
        return [graphs[i] for i in idxs], [labels[i] for i in idxs]

    train_graphs, train_labels = pick(train_idx)
    val_graphs, val_labels = pick(val_idx)
    test_graphs, test_labels = pick(test_idx)

    logging.getLogger(__name__).info(f"Dataset split:")
    logging.getLogger(__name__).info(f"  Train: {len(train_graphs)} samples (malware: {sum(train_labels)}, benign: {len(train_labels)-sum(train_labels)})")
    logging.getLogger(__name__).info(f"  Val: {len(val_graphs)} samples (malware: {sum(val_labels)}, benign: {len(val_labels)-sum(val_labels)})")
    logging.getLogger(__name__).info(f"  Test: {len(test_graphs)} samples (malware: {sum(test_labels)}, benign: {len(test_labels)-sum(test_labels)})")

    return (train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels)


def main(args):
    # Use DEBUG if user asked for verbose via environment or you can change here
    logger = setup_logging(args.log_dir, level=logging.INFO)
    logger.info("=" * 80)
    logger.info("APK Preprocessing Pipeline")
    logger.info("=" * 80)

    logger.info("Initializing preprocessing components...")
    api_filter = SensitiveAPIFilter(args.sensitive_apis, aggressive_matching=args.aggressive_matching)
    graph_builder = APICallGraphBuilder(api_filter)

    reports_dir = Path(args.output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("Processing Malware APKs")
    logger.info("=" * 80)
    malware_graphs, malware_labels = process_apk_directory(
        args.malware_dir, label=1,
        api_filter=api_filter,
        graph_builder=graph_builder,
        n_workers=args.n_workers,
        reports_dir=reports_dir
    )

    logger.info("\n" + "=" * 80)
    logger.info("Processing Benign APKs")
    logger.info("=" * 80)
    benign_graphs, benign_labels = process_apk_directory(
        args.benign_dir, label=0,
        api_filter=api_filter,
        graph_builder=graph_builder,
        n_workers=args.n_workers,
        reports_dir=reports_dir
    )

    all_graphs = malware_graphs + benign_graphs
    all_labels = malware_labels + benign_labels

    logger.info(f"\nTotal processed: {len(all_graphs)} APKs")
    logger.info(f"  Malware: {sum(all_labels)}")
    logger.info(f"  Benign: {len(all_labels) - sum(all_labels)}")

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

    logger.info("\n" + "=" * 80)
    logger.info("Saving Processed Datasets")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(train_graphs, train_labels, str(train_dir / 'graphs.pkl'))

    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(val_graphs, val_labels, str(val_dir / 'graphs.pkl'))

    test_dir = output_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    MalwareGraphDataset.save_dataset(test_graphs, test_labels, str(test_dir / 'graphs.pkl'))

    logger.info(f"Datasets saved to {output_dir}")

    # Safe stats
    stats = {
        'total_apks': len(all_graphs),
        'malware_count': sum(all_labels) if len(all_labels) > 0 else 0,
        'benign_count': (len(all_labels) - sum(all_labels)) if len(all_labels) > 0 else 0,
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
    }

    try:
        stats['avg_nodes'] = float(np.mean([g.number_of_nodes() for g in all_graphs])) if len(all_graphs) > 0 else 0.0
        stats['avg_edges'] = float(np.mean([g.number_of_edges() for g in all_graphs])) if len(all_graphs) > 0 else 0.0
    except Exception:
        stats['avg_nodes'] = 0.0
        stats['avg_edges'] = 0.0

    stats_path = output_dir / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to {stats_path}")

    # Optional: aggregate category distribution for all processed APKs from reports
    try:
        aggregated = {}
        reports = list(reports_dir.glob('*.json'))
        for rpt in reports:
            try:
                with open(rpt, 'r') as rf:
                    data = json.load(rf)
                    if 'category_stats' in data:
                        for k, v in data['category_stats'].items():
                            aggregated[k] = aggregated.get(k, 0) + v
            except Exception:
                continue
        if aggregated and _HAS_MPL:
            viz_path = output_dir / 'all_api_categories.png'
            visualize_api_distribution(aggregated, viz_path, title="Aggregate API Category Distribution")
    except Exception:
        pass

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
                       help='Number of parallel workers (not used in seq mode)')
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
    parser.add_argument('--aggressive_matching', action='store_true',
                       help='Enable aggressive method-only matching (useful for research, may increase false positives)')
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    main(args)
