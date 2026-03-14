"""
Inference Script for XAIDroid
Complete pipeline: APK → Prediction → Code Localization
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
import torch
import tempfile
import shutil
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.apk_decompiler import APKDecompiler
from src.preprocessing.api_extractor import SensitiveAPIFilter
from src.preprocessing.graph_builder import APICallGraphBuilder
from src.dataset.graph_dataset import MalwareGraphDataset
from src.models.ensemble import XAIDroidEnsemble
from src.explainability.code_localizer import CodeLocalizer, EnsembleLocalizer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_models(gat_path: str, gam_path: str, device: str):
    """Load trained models."""
    gat_model, gam_model = XAIDroidEnsemble.load_models(
        gat_path, gam_path, device
    )
    return gat_model, gam_model


def preprocess_apk(apk_path: str, api_filter, graph_builder, logger) -> Dict:
    """
    Preprocess single APK for inference.

    Args:
        apk_path: Path to APK file
        api_filter: SensitiveAPIFilter instance
        graph_builder: APICallGraphBuilder instance
        logger: Logger instance

    Returns:
        Dictionary with preprocessed data
    """
    logger.info(f"Preprocessing APK: {apk_path}")

    # FIX IN6: use a temp dir for smali output so inference doesn't depend
    # on data/processed/smali existing or having write permissions.
    # Mirrors how preprocess_apks.py passes an explicit output_dir.
    smali_tmp = tempfile.mkdtemp()
    try:
        decompiler = APKDecompiler(output_dir=smali_tmp)
        decompiled = decompiler.decompile(apk_path)
    finally:
        shutil.rmtree(smali_tmp, ignore_errors=True)

    if not decompiled["success"]:
        raise ValueError(f"Failed to decompile APK: {decompiled.get('error')}")

    # FIX IN1: api_filter was passed in but never called, so the graph was
    # built from all raw API calls (10,000+) instead of the 50-200 sensitive
    # ones the model was trained on — a train/inference distribution mismatch.
    # Apply the same filtering step that preprocess_apks.py uses.
    # FIX REFLECTION: pass full decompiled dict (not just decompiled["api_calls"])
    # so that reflection_evidence is also forwarded to filter_api_calls().
    # Passing only api_calls silently drops all reflection-based sensitive API hits.
    filtered = api_filter.filter_api_calls(decompiled)
    decompiled["api_calls"] = filtered["methods"]

    # Build graph
    decompiled["label"] = "unknown"
    graph = graph_builder.build_graph(decompiled)

    if graph.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes")

    logger.info(f"Graph built: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges")

    # Convert to PyG data
    node_names = list(graph.nodes())

    # FIX IN2: hardcoded root="temp" caused PyG to write data_0.pt on the
    # first call and then silently return that stale file for every subsequent
    # APK, because PyG skips process() when the file already exists.
    # Use a fresh temporary directory that is cleaned up after loading.
    tmp_dir = tempfile.mkdtemp()
    try:
        dataset = MalwareGraphDataset(
            root=tmp_dir,
            graphs=[graph],
            labels=[0]  # Dummy label
        )
        data = dataset[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "data": data,
        "graph": graph,
        "node_names": node_names,
        "apk_name": decompiled["apk_name"],
        "package_name": decompiled["package_name"],
        "num_methods": len(decompiled["methods"]),
        "num_classes": len(decompiled["classes"])
    }


def run_inference(
    preprocessed: Dict,
    ensemble: XAIDroidEnsemble,
    localizer: EnsembleLocalizer,
    device: str,
    logger
) -> Dict:
    """
    Run inference with code localization.

    Args:
        preprocessed: Preprocessed APK data
        ensemble: XAIDroid ensemble model
        localizer: Code localizer
        device: Device to run on
        logger: Logger instance

    Returns:
        Complete inference result
    """
    logger.info("Running inference...")

    data = preprocessed["data"].to(device)

    # Create batch tensor (single sample)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    # Get predictions
    # FIX: extract graph_size added by graph_dataset.py and forward it so
    # gat_model receives the size features it was trained with.
    graph_size = getattr(data, "graph_size", None)
    if graph_size is not None:
        graph_size = graph_size.unsqueeze(0)  # [1, 2] — single sample batch dim

    prediction_result = ensemble.predict(
        data.x, data.edge_index, batch, graph_size=graph_size, return_details=True
    )

    logger.info(f"Ensemble prediction: {prediction_result['ensemble_prediction'][0]}")
    logger.info(f"Confidence: {prediction_result['ensemble_confidence'][0]:.4f}")
    logger.info(f"GAT prediction: {prediction_result['gat_prediction'][0]}")
    logger.info(f"GAM prediction: {prediction_result['gam_prediction'][0]}")
    logger.info(f"Models agree: {bool(prediction_result['agreement'][0])}")

    # If predicted as malware, perform code localization
    if prediction_result['ensemble_prediction'][0] == 1:
        logger.info("\nPerforming malicious code localization...")

        # Extract attention from both models
        gat_attention = ensemble.gat_model.extract_node_attention(
            data.x, data.edge_index, batch, graph_size=graph_size
        )
        gam_attention = ensemble.gam_model.extract_node_importance(
            data.x, data.edge_index, batch
        )

        # Localize malicious code
        # FIX IN4: pass data (not None) as 4th argument, matching the
        # signature used in ensemble.predict_with_localization() after
        # that method was fixed to forward data consistently.
        localization = localizer.localize(
            gat_attention,
            gam_attention,
            preprocessed["node_names"],
            data
        )

        logger.info(f"Found {len(localization['malicious_methods'])} malicious methods")
        logger.info(f"Found {len(localization['malicious_classes'])} malicious classes")
    else:
        localization = None
        logger.info("\nPredicted as benign - no code localization performed")

    # Build result
    result = {
        "apk_name": preprocessed["apk_name"],
        "package_name": preprocessed["package_name"],
        "apk_stats": {
            "num_methods": preprocessed["num_methods"],
            "num_classes": preprocessed["num_classes"],
            "graph_nodes": data.num_nodes,
            "graph_edges": data.edge_index.shape[1]
        },
        "prediction": {
            "ensemble": {
                "label": "malware" if prediction_result['ensemble_prediction'][0] == 1 else "benign",
                "confidence": float(prediction_result['ensemble_confidence'][0])
            },
            "gat": {
                "label": "malware" if prediction_result['gat_prediction'][0] == 1 else "benign",
                "confidence": float(prediction_result['gat_confidence'][0])
            },
            "gam": {
                "label": "malware" if prediction_result['gam_prediction'][0] == 1 else "benign",
                "confidence": float(prediction_result['gam_confidence'][0])
            },
            "models_agree": bool(prediction_result['agreement'][0]),
            "ensemble_rule": prediction_result['ensemble_rule']
        },
        "localization": None
    }

    # Add localization if malware
    if localization:
        result["localization"] = {
            "malicious_methods_count": len(localization['malicious_methods']),
            "malicious_classes_count": len(localization['malicious_classes']),
            "top_malicious_methods": [
                {
                    "method": method,
                    "score": float(score)
                }
                for method, score in localization['malicious_methods'][:10]
            ],
            "malicious_classes": [
                {
                    "class_name": class_name,
                    "score": float(score),
                    "malicious_methods": [
                        m.split("->")[1] if "->" in m else m
                        for m in methods[:5]
                    ]
                }
                for class_name, score, methods in localization['malicious_classes']
            ]
        }

    return result


def format_result(result: Dict) -> str:
    """Format inference result as human-readable string."""
    output = []

    output.append("=" * 80)
    output.append("XAIDROID MALWARE DETECTION RESULT")
    output.append("=" * 80)

    # APK Info
    output.append(f"\nAPK Information:")
    output.append(f"  Name: {result['apk_name']}")
    output.append(f"  Package: {result['package_name']}")
    output.append(f"  Methods: {result['apk_stats']['num_methods']}")
    output.append(f"  Classes: {result['apk_stats']['num_classes']}")
    output.append(f"  Graph Nodes: {result['apk_stats']['graph_nodes']}")
    output.append(f"  Graph Edges: {result['apk_stats']['graph_edges']}")

    # Prediction
    output.append(f"\n{'=' * 80}")
    pred = result['prediction']
    ensemble_label = pred['ensemble']['label'].upper()
    output.append(f"PREDICTION: {ensemble_label}")
    output.append(f"Confidence: {pred['ensemble']['confidence']:.2%}")
    output.append(f"{'=' * 80}")

    output.append(f"\nModel Predictions:")
    output.append(f"  GAT:  {pred['gat']['label']:8s}  (confidence: {pred['gat']['confidence']:.2%})")
    output.append(f"  GAM:  {pred['gam']['label']:8s}  (confidence: {pred['gam']['confidence']:.2%})")
    output.append(f"  Ensemble Rule: {pred['ensemble_rule']}")
    output.append(f"  Models Agree: {'Yes' if pred['models_agree'] else 'No'}")

    # Localization
    if result['localization']:
        loc = result['localization']
        output.append(f"\n{'=' * 80}")
        output.append("MALICIOUS CODE LOCALIZATION")
        output.append("=" * 80)

        output.append(f"\nSummary:")
        output.append(f"  Malicious Classes: {loc['malicious_classes_count']}")
        output.append(f"  Malicious Methods: {loc['malicious_methods_count']}")

        if loc['malicious_classes']:
            output.append(f"\nMalicious Classes:")
            for i, cls in enumerate(loc['malicious_classes'], 1):
                output.append(f"\n  [{i}] {cls['class_name']}")
                output.append(f"      Score: {cls['score']:.6f}")
                output.append(f"      Top Methods:")
                for method in cls['malicious_methods']:
                    output.append(f"        ✗ {method}")

        if loc['top_malicious_methods']:
            output.append(f"\nTop Malicious Methods (All Classes):")
            for i, method_info in enumerate(loc['top_malicious_methods'][:5], 1):
                output.append(f"  [{i}] {method_info['method']}")
                output.append(f"      Score: {method_info['score']:.6f}")

    output.append("\n" + "=" * 80)

    return "\n".join(output)


def main(args):
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("XAIDroid Inference")
    logger.info("=" * 80)

    # FIX IN5: Load config so ensemble_rule and thresholds are read from
    # config.yaml instead of being hardcoded. CLI args override config values
    # when explicitly provided.
    config = load_config(args.config)
    explainability_cfg = config.get('explainability', {})

    ensemble_rule    = explainability_cfg.get('ensemble_rule', 'AND')
    method_threshold = args.method_threshold if args.method_threshold is not None \
                       else explainability_cfg.get('method_threshold', 0.0001)
    class_threshold  = args.class_threshold if args.class_threshold is not None \
                       else explainability_cfg.get('class_threshold', 0.001)

    logger.info(f"Ensemble rule:    {ensemble_rule}")
    logger.info(f"Method threshold: {method_threshold}")
    logger.info(f"Class threshold:  {class_threshold}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    logger.info("\nLoading models...")
    gat_model, gam_model = load_models(args.gat_model, args.gam_model, device)

    # FIX IN5 (cont): ensemble_rule now comes from config, not hardcoded "AND"
    ensemble = XAIDroidEnsemble(
        gat_model=gat_model,
        gam_model=gam_model,
        ensemble_rule=ensemble_rule,
        device=device
    )

    # Initialize code localizer
    # FIX IN5 (cont): thresholds and ensemble_rule from config
    gat_localizer = CodeLocalizer(
        method_threshold=method_threshold,
        class_threshold=class_threshold
    )
    gam_localizer = CodeLocalizer(
        method_threshold=method_threshold,
        class_threshold=class_threshold
    )
    localizer = EnsembleLocalizer(gat_localizer, gam_localizer, ensemble_rule=ensemble_rule)

    # Initialize preprocessing components
    sensitive_apis_path = args.sensitive_apis or config['paths']['sensitive_apis']
    logger.info("\nInitializing preprocessing...")
    api_filter = SensitiveAPIFilter(sensitive_apis_path)
    graph_builder = APICallGraphBuilder(api_filter)

    # Preprocess APK
    try:
        preprocessed = preprocess_apk(
            args.apk_path, api_filter, graph_builder, logger
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # Run inference
    try:
        result = run_inference(
            preprocessed, ensemble, localizer, device, logger
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

    # Save result
    output_path = Path(args.output) / f"{Path(args.apk_path).stem}_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResult saved to: {output_path}")

    # Print formatted result
    print("\n" + format_result(result))

    logger.info("\nInference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAIDroid inference on APK")
    parser.add_argument('--apk_path', type=str, required=True,
                        help='Path to APK file')
    parser.add_argument('--gat_model', type=str, required=True,
                        help='Path to trained GAT model')
    parser.add_argument('--gam_model', type=str, required=True,
                        help='Path to trained GAM model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--sensitive_apis', type=str, default=None,
                        help='Path to sensitive APIs JSON (overrides config)')
    # FIX IN5: default=None so we can detect whether the user explicitly passed
    # a value. If not passed, the value is read from config.yaml instead.
    parser.add_argument('--method_threshold', type=float, default=None,
                        help='Method localization threshold (overrides config)')
    parser.add_argument('--class_threshold', type=float, default=None,
                        help='Class localization threshold (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()
    main(args)
