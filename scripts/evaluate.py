"""
Evaluation Script for XAIDroid
Comprehensive evaluation of GAT, GAM, and Ensemble models
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import XAIDroidEnsemble
from src.dataset.graph_dataset import MalwareGraphDataset


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def evaluate_model(model, data_loader, device, model_name, logger):
    """
    Evaluate a single model.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device to use
        model_name: Name for logging
        logger: Logger instance
        
    Returns:
        Dictionary of metrics and predictions
    """
    logger.info(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Malware class probability
    
    # Calculate metrics
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }
    
    # Calculate per-class metrics
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Log results
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    
    return metrics, preds, labels, probs


def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malware'],
                yticklabels=['Benign', 'Malware'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'confusion_matrix_{model_name.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_roc_curves(results, output_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
        auc = data['metrics']['auc_roc']
        plt.plot(fpr, tpr, linewidth=2,
                label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - XAIDroid Models', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_metrics_comparison(results, output_dir):
    """Plot bar chart comparing all metrics across models."""
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 4))
    
    for idx, metric_name in enumerate(metrics_names):
        values = [results[model]['metrics'][metric_name] for model in models]
        
        bars = axes[idx].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[idx].set_ylim([0.8, 1.0])  # Adjust based on expected performance
        axes[idx].set_title(metric_name.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.4f}',
                          ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_report(results, output_dir, logger):
    """Generate comprehensive evaluation report."""
    report_lines = []
    
    report_lines.append("=" * 100)
    report_lines.append("XAIDROID EVALUATION REPORT")
    report_lines.append("=" * 100)
    
    # Summary table
    report_lines.append("\nPerformance Summary:")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} "
                       f"{'F1 Score':<12} {'AUC-ROC':<12}")
    report_lines.append("-" * 100)
    
    for model_name, data in results.items():
        m = data['metrics']
        report_lines.append(
            f"{model_name:<15} "
            f"{m['accuracy']:<12.4f} "
            f"{m['precision']:<12.4f} "
            f"{m['recall']:<12.4f} "
            f"{m['f1_score']:<12.4f} "
            f"{m['auc_roc']:<12.4f}"
        )
    
    # Detailed metrics per model
    for model_name, data in results.items():
        report_lines.append("\n" + "=" * 100)
        report_lines.append(f"{model_name} - Detailed Metrics")
        report_lines.append("=" * 100)
        
        m = data['metrics']
        cm = np.array(m['confusion_matrix'])
        
        report_lines.append(f"\nConfusion Matrix:")
        report_lines.append(f"                 Predicted Benign    Predicted Malware")
        report_lines.append(f"Actual Benign         {cm[0][0]:<10}      {cm[0][1]:<10}")
        report_lines.append(f"Actual Malware        {cm[1][0]:<10}      {cm[1][1]:<10}")
        
        report_lines.append(f"\nClassification Metrics:")
        report_lines.append(f"  True Positives:  {m['true_positives']}")
        report_lines.append(f"  True Negatives:  {m['true_negatives']}")
        report_lines.append(f"  False Positives: {m['false_positives']}")
        report_lines.append(f"  False Negatives: {m['false_negatives']}")
        report_lines.append(f"  Specificity:     {m['specificity']:.4f}")
    
    # Model comparison
    report_lines.append("\n" + "=" * 100)
    report_lines.append("MODEL COMPARISON")
    report_lines.append("=" * 100)
    
    # Find best model per metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        best_model = max(results.keys(), 
                        key=lambda m: results[m]['metrics'][metric])
        best_value = results[best_model]['metrics'][metric]
        report_lines.append(f"\nBest {metric.replace('_', ' ').title()}: "
                          f"{best_model} ({best_value:.4f})")
    
    # Agreement analysis (if ensemble)
    if 'Ensemble' in results:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("ENSEMBLE ANALYSIS")
        report_lines.append("=" * 100)
        
        # Compare ensemble with individual models
        ensemble_f1 = results['Ensemble']['metrics']['f1_score']
        gat_f1 = results['GAT']['metrics']['f1_score']
        gam_f1 = results['GAM']['metrics']['f1_score']
        
        improvement_gat = ((ensemble_f1 - gat_f1) / gat_f1) * 100
        improvement_gam = ((ensemble_f1 - gam_f1) / gam_f1) * 100
        
        report_lines.append(f"\nEnsemble Improvement:")
        report_lines.append(f"  vs GAT: {improvement_gat:+.2f}%")
        report_lines.append(f"  vs GAM: {improvement_gam:+.2f}%")
    
    report_lines.append("\n" + "=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = Path(output_dir) / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # Print report
    print("\n" + report_text)
    
    return str(report_path)


def main(args):
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("XAIDroid Model Evaluation")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test data
    logger.info("\nLoading test dataset...")
    test_graphs, test_labels = MalwareGraphDataset.load_dataset(
        args.test_data + '/graphs.pkl'
    )
    
    test_dataset = MalwareGraphDataset(
        root=args.test_data,
        graphs=test_graphs,
        labels=test_labels
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Test set: {len(test_dataset)} samples")
    
    # Load models
    logger.info("\nLoading models...")
    gat_model, gam_model = XAIDroidEnsemble.load_models(
        args.gat_model, args.gam_model, device
    )
    
    # Initialize ensemble
    ensemble = XAIDroidEnsemble(
        gat_model=gat_model,
        gam_model=gam_model,
        ensemble_rule="AND",
        device=device
    )
    
    # Evaluate each model
    results = {}
    
    # Evaluate GAT
    gat_metrics, gat_preds, gat_labels, gat_probs = evaluate_model(
        gat_model, test_loader, device, "GAT", logger
    )
    results['GAT'] = {
        'metrics': gat_metrics,
        'preds': gat_preds,
        'labels': gat_labels,
        'probs': gat_probs
    }
    
    # Evaluate GAM
    gam_metrics, gam_preds, gam_labels, gam_probs = evaluate_model(
        gam_model, test_loader, device, "GAM", logger
    )
    results['GAM'] = {
        'metrics': gam_metrics,
        'preds': gam_preds,
        'labels': gam_labels,
        'probs': gam_probs
    }
    
    # Evaluate Ensemble
    logger.info("\nEvaluating Ensemble...")
    ensemble_metrics = ensemble.evaluate(test_loader)
    
    # Extract ensemble predictions for plotting
    all_ensemble_preds = []
    all_ensemble_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            result = ensemble.predict(
                batch.x, batch.edge_index, batch.batch, return_details=True
            )
            all_ensemble_preds.extend(result['ensemble_prediction'])
            all_ensemble_probs.extend(result['ensemble_confidence'])
            all_labels.extend(batch.y.cpu().numpy())
    
    results['Ensemble'] = {
        'metrics': ensemble_metrics['ensemble'],
        'preds': np.array(all_ensemble_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_ensemble_probs)
    }
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion matrices
    for model_name, data in results.items():
        cm = np.array(data['metrics']['confusion_matrix'])
        cm_path = plot_confusion_matrix(cm, model_name, output_dir)
        logger.info(f"  Confusion matrix saved: {cm_path}")
    
    # ROC curves
    roc_path = plot_roc_curves(results, output_dir)
    logger.info(f"  ROC curves saved: {roc_path}")
    
    # Metrics comparison
    metrics_path = plot_metrics_comparison(results, output_dir)
    logger.info(f"  Metrics comparison saved: {metrics_path}")
    
    # Generate report
    logger.info("\nGenerating evaluation report...")
    report_path = generate_report(results, output_dir, logger)
    logger.info(f"  Report saved: {report_path}")
    
    # Save results as JSON
    results_json = {
        model: {
            'metrics': data['metrics']
        }
        for model, data in results.items()
    }
    
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"  JSON results saved: {json_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XAIDroid models")
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--gat_model', type=str, required=True,
                       help='Path to trained GAT model')
    parser.add_argument('--gam_model', type=str, required=True,
                       help='Path to trained GAM model')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    main(args)