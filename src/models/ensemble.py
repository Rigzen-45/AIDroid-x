"""
Ensemble Module for XAIDroid
Combines GAT and GAM predictions using AND/OR rules
Paper uses AND rule: GAM ∧ GAT (both must agree for malware classification)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class XAIDroidEnsemble:
    """
    Ensemble classifier combining GAT and GAM models.
    Implements configurable voting rules for final prediction.
    """
    
    def __init__(
        self,
        gat_model,
        gam_model,
        ensemble_rule: str = "AND",
        gat_weight: float = 0.5,
        gam_weight: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize ensemble model.
        
        Args:
            gat_model: Trained GAT model
            gam_model: Trained GAM model
            ensemble_rule: Voting strategy - "AND", "OR", "WEIGHTED", "MAJORITY"
            gat_weight: Weight for GAT in weighted voting
            gam_weight: Weight for GAM in weighted voting
            device: Device to run models on
        """
        self.gat_model = gat_model.to(device)
        self.gam_model = gam_model.to(device)
        self.ensemble_rule = ensemble_rule.upper()
        self.gat_weight = gat_weight
        self.gam_weight = gam_weight
        self.device = device
        
        # Set models to eval mode
        self.gat_model.eval()
        self.gam_model.eval()
        
        logger.info(f"Initialized XAIDroid Ensemble:")
        logger.info(f"  Rule: {ensemble_rule}")
        logger.info(f"  Weights: GAT={gat_weight}, GAM={gam_weight}")
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_details: bool = False
    ) -> Dict:
        """
        Make ensemble prediction.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            return_details: If True, return detailed results from both models
            
        Returns:
            Dictionary containing:
                - ensemble_prediction: Final prediction (0=benign, 1=malware)
                - ensemble_confidence: Confidence score [0, 1]
                - gat_prediction: GAT prediction
                - gat_confidence: GAT confidence
                - gam_prediction: GAM prediction
                - gam_confidence: GAM confidence
                - agreement: Whether models agree
                - attention_scores: Attention from both models (if return_details)
        """
        with torch.no_grad():
            # Get predictions from both models
            gat_logits, gat_attention = self.gat_model(
                x, edge_index, batch, return_attention=True
            )
            gam_logits = self.gam_model(x, edge_index, batch)
            
            # Convert to probabilities
            gat_probs = F.softmax(gat_logits, dim=1)
            gam_probs = F.softmax(gam_logits, dim=1)
            
            # Get individual predictions (class 1 = malware)
            gat_pred = torch.argmax(gat_probs, dim=1)
            gam_pred = torch.argmax(gam_probs, dim=1)
            
            # Get confidence scores for malware class
            gat_confidence = gat_probs[:, 1]
            gam_confidence = gam_probs[:, 1]
            
            # Apply ensemble rule
            if self.ensemble_rule == "AND":
                ensemble_pred, ensemble_conf = self._predict_and(
                    gat_pred, gam_pred, gat_confidence, gam_confidence
                )
            elif self.ensemble_rule == "OR":
                ensemble_pred, ensemble_conf = self._predict_or(
                    gat_pred, gam_pred, gat_confidence, gam_confidence
                )
            elif self.ensemble_rule == "WEIGHTED":
                ensemble_pred, ensemble_conf = self._predict_weighted(
                    gat_probs, gam_probs
                )
            elif self.ensemble_rule == "MAJORITY":
                ensemble_pred, ensemble_conf = self._predict_majority(
                    gat_pred, gam_pred, gat_confidence, gam_confidence
                )
            else:
                raise ValueError(f"Unknown ensemble rule: {self.ensemble_rule}")
            
            # Check agreement
            agreement = (gat_pred == gam_pred).float()
            
            # Build result dictionary
            result = {
                "ensemble_prediction": ensemble_pred.cpu().numpy(),
                "ensemble_confidence": ensemble_conf.cpu().numpy(),
                "gat_prediction": gat_pred.cpu().numpy(),
                "gat_confidence": gat_confidence.cpu().numpy(),
                "gam_prediction": gam_pred.cpu().numpy(),
                "gam_confidence": gam_confidence.cpu().numpy(),
                "agreement": agreement.cpu().numpy(),
                "ensemble_rule": self.ensemble_rule
            }
            
            # Add detailed information if requested
            if return_details:
                result["gat_logits"] = gat_logits.cpu().numpy()
                result["gam_logits"] = gam_logits.cpu().numpy()
                result["gat_probs"] = gat_probs.cpu().numpy()
                result["gam_probs"] = gam_probs.cpu().numpy()
                result["gat_attention"] = gat_attention
        
        return result
    
    def _predict_and(
        self,
        gat_pred: torch.Tensor,
        gam_pred: torch.Tensor,
        gat_conf: torch.Tensor,
        gam_conf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AND rule: Both models must predict malware.
        Paper's default rule: GAM ∧ GAT
        
        Args:
            gat_pred: GAT predictions
            gam_pred: GAM predictions
            gat_conf: GAT confidence
            gam_conf: GAM confidence
            
        Returns:
            Tuple of (predictions, confidence)
        """
        # Both must predict 1 (malware)
        ensemble_pred = (gat_pred == 1) & (gam_pred == 1)
        ensemble_pred = ensemble_pred.long()
        
        # Confidence: minimum of both (conservative)
        ensemble_conf = torch.min(gat_conf, gam_conf)
        
        return ensemble_pred, ensemble_conf
    
    def _predict_or(
        self,
        gat_pred: torch.Tensor,
        gam_pred: torch.Tensor,
        gat_conf: torch.Tensor,
        gam_conf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        OR rule: Either model predicting malware is sufficient.
        
        Args:
            gat_pred: GAT predictions
            gam_pred: GAM predictions
            gat_conf: GAT confidence
            gam_conf: GAM confidence
            
        Returns:
            Tuple of (predictions, confidence)
        """
        # Either predicts 1 (malware)
        ensemble_pred = (gat_pred == 1) | (gam_pred == 1)
        ensemble_pred = ensemble_pred.long()
        
        # Confidence: maximum of both
        ensemble_conf = torch.max(gat_conf, gam_conf)
        
        return ensemble_pred, ensemble_conf
    
    def _predict_weighted(
        self,
        gat_probs: torch.Tensor,
        gam_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted voting: Combine probabilities with weights.
        
        Args:
            gat_probs: GAT probabilities [batch_size, 2]
            gam_probs: GAM probabilities [batch_size, 2]
            
        Returns:
            Tuple of (predictions, confidence)
        """
        # Weighted combination of probabilities
        ensemble_probs = (self.gat_weight * gat_probs + 
                         self.gam_weight * gam_probs)
        
        # Normalize
        ensemble_probs = ensemble_probs / (self.gat_weight + self.gam_weight)
        
        # Get predictions
        ensemble_pred = torch.argmax(ensemble_probs, dim=1)
        ensemble_conf = ensemble_probs[:, 1]
        
        return ensemble_pred, ensemble_conf
    
    def _predict_majority(
        self,
        gat_pred: torch.Tensor,
        gam_pred: torch.Tensor,
        gat_conf: torch.Tensor,
        gam_conf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Majority voting: Use model with higher confidence.
        
        Args:
            gat_pred: GAT predictions
            gam_pred: GAM predictions
            gat_conf: GAT confidence
            gam_conf: GAM confidence
            
        Returns:
            Tuple of (predictions, confidence)
        """
        # Use prediction from more confident model
        use_gat = gat_conf > gam_conf
        
        ensemble_pred = torch.where(use_gat, gat_pred, gam_pred)
        ensemble_conf = torch.where(use_gat, gat_conf, gam_conf)
        
        return ensemble_pred, ensemble_conf
    
    def predict_with_localization(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_names: List[str],
        code_localizer
    ) -> Dict:
        """
        Make prediction with malicious code localization.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            node_names: Node identifiers
            code_localizer: CodeLocalizer or EnsembleLocalizer instance
            
        Returns:
            Dictionary with predictions and localization
        """
        # Get ensemble prediction
        result = self.predict(x, edge_index, batch, return_details=True)
        
        # Extract attention scores
        gat_node_attention = self.gat_model.extract_node_attention(
            x, edge_index, batch
        )
        gam_node_attention = self.gam_model.extract_node_importance(
            x, edge_index, batch
        )
        
        # Perform code localization
        localization = code_localizer.localize(
            gat_node_attention,
            gam_node_attention,
            node_names,
            None
        )
        
        # Combine results
        result["localization"] = localization
        
        return result
    
    def evaluate(self, test_loader) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            test_loader: PyG DataLoader
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_ensemble_preds = []
        all_gat_preds = []
        all_gam_preds = []
        all_labels = []
        all_agreement = []
        
        self.gat_model.eval()
        self.gam_model.eval()
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Get predictions
                result = self.predict(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.batch,
                    return_details=False
                )
                
                all_ensemble_preds.extend(result["ensemble_prediction"])
                all_gat_preds.extend(result["gat_prediction"])
                all_gam_preds.extend(result["gam_prediction"])
                all_labels.extend(batch_data.y.cpu().numpy())
                all_agreement.extend(result["agreement"])
        
        # Convert to numpy arrays
        ensemble_preds = np.array(all_ensemble_preds)
        gat_preds = np.array(all_gat_preds)
        gam_preds = np.array(all_gam_preds)
        labels = np.array(all_labels)
        agreement = np.array(all_agreement)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        metrics = {
            "ensemble": {
                "accuracy": accuracy_score(labels, ensemble_preds),
                "precision": precision_score(labels, ensemble_preds, zero_division=0),
                "recall": recall_score(labels, ensemble_preds, zero_division=0),
                "f1_score": f1_score(labels, ensemble_preds, zero_division=0),
                "confusion_matrix": confusion_matrix(labels, ensemble_preds).tolist()
            },
            "gat": {
                "accuracy": accuracy_score(labels, gat_preds),
                "precision": precision_score(labels, gat_preds, zero_division=0),
                "recall": recall_score(labels, gat_preds, zero_division=0),
                "f1_score": f1_score(labels, gat_preds, zero_division=0)
            },
            "gam": {
                "accuracy": accuracy_score(labels, gam_preds),
                "precision": precision_score(labels, gam_preds, zero_division=0),
                "recall": recall_score(labels, gam_preds, zero_division=0),
                "f1_score": f1_score(labels, gam_preds, zero_division=0)
            },
            "agreement_rate": float(agreement.mean()),
            "ensemble_rule": self.ensemble_rule
        }
        
        # Log results
        logger.info("=" * 60)
        logger.info("Ensemble Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Ensemble ({self.ensemble_rule}):")
        logger.info(f"  Accuracy:  {metrics['ensemble']['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['ensemble']['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['ensemble']['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['ensemble']['f1_score']:.4f}")
        logger.info(f"\nGAT Only:")
        logger.info(f"  Accuracy:  {metrics['gat']['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['gat']['f1_score']:.4f}")
        logger.info(f"\nGAM Only:")
        logger.info(f"  Accuracy:  {metrics['gam']['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['gam']['f1_score']:.4f}")
        logger.info(f"\nModel Agreement: {metrics['agreement_rate']:.2%}")
        logger.info("=" * 60)
        
        return metrics
    
    def save_models(self, gat_path: str, gam_path: str):
        """
        Save both models.
        
        Args:
            gat_path: Path to save GAT model
            gam_path: Path to save GAM model
        """
        torch.save({
            "model_state_dict": self.gat_model.state_dict(),
            "model_config": {
                "num_node_features": self.gat_model.num_node_features,
                "hidden_units": self.gat_model.hidden_units,
                "num_heads": self.gat_model.num_heads,
                "num_classes": self.gat_model.num_classes,
                "dropout": self.gat_model.dropout
            }
        }, gat_path)
        
        torch.save({
            "model_state_dict": self.gam_model.state_dict(),
            "model_config": {
                "num_node_features": self.gam_model.encoder.conv1.in_channels
            }
        }, gam_path)
        
        logger.info(f"Saved GAT model to {gat_path}")
        logger.info(f"Saved GAM model to {gam_path}")
    
    @staticmethod
    def load_models(gat_path: str, gam_path: str, device: str = "cuda"):
        """
        Load both models from checkpoints.
        
        Args:
            gat_path: Path to GAT checkpoint
            gam_path: Path to GAM checkpoint
            device: Device to load models on
            
        Returns:
            Tuple of (gat_model, gam_model)
        """
        from src.models.gat_model import GAT
        from src.models.gam_model import GAMClassifier
        
        # Load GAT
        gat_checkpoint = torch.load(gat_path, map_location=device)
        gat_config = gat_checkpoint["model_config"]
        gat_model = GAT(**gat_config)
        gat_model.load_state_dict(gat_checkpoint["model_state_dict"])
        gat_model.to(device)
        gat_model.eval()
        
        # Load GAM
        gam_checkpoint = torch.load(gam_path, map_location=device)
        gam_config = gam_checkpoint["model_config"]
        gam_model = GAMClassifier(**gam_config)
        gam_model.load_state_dict(gam_checkpoint["model_state_dict"])
        gam_model.to(device)
        gam_model.eval()
        
        logger.info(f"Loaded GAT model from {gat_path}")
        logger.info(f"Loaded GAM model from {gam_path}")
        
        return gat_model, gam_model


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from src.models.gat_model import GAT
    from src.models.gam_model import GAMClassifier
    
    # Create dummy models
    gat_model = GAT(num_node_features=15, hidden_units=8, num_heads=8)
    gam_model = GAMClassifier(num_node_features=15)
    
    # Initialize ensemble
    ensemble = XAIDroidEnsemble(
        gat_model=gat_model,
        gam_model=gam_model,
        ensemble_rule="AND",
        device="cpu"
    )
    
    # Test prediction
    batch_size = 2
    num_nodes = 50
    num_edges = 100
    
    x = torch.randn(num_nodes, 15)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.cat([torch.zeros(25, dtype=torch.long),
                      torch.ones(25, dtype=torch.long)])
    
    result = ensemble.predict(x, edge_index, batch, return_details=True)
    
    print("\nEnsemble Prediction:")
    print(f"  Ensemble: {result['ensemble_prediction']} "
          f"(conf: {result['ensemble_confidence']})")
    print(f"  GAT: {result['gat_prediction']} "
          f"(conf: {result['gat_confidence']})")
    print(f"  GAM: {result['gam_prediction']} "
          f"(conf: {result['gam_confidence']})")
    print(f"  Agreement: {result['agreement']}")
    print(f"  Rule: {result['ensemble_rule']}")