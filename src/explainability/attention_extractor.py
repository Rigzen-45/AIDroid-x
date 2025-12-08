"""
Attention Extractor Module
Extracts and processes attention scores from GAT and GAM models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionExtractor:
    """
    Extracts attention scores from graph neural networks.
    Supports GAT and GAM models with multi-head attention.
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Initialize attention extractor.
        
        Args:
            model: Trained model (GAT or GAM)
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def extract_gat_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention scores from GAT model.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Dictionary containing:
                - layer1_attention: Attention from first GAT layer
                - layer2_attention: Attention from second GAT layer
                - node_attention: Aggregated node-level attention
                - edge_attention: Edge-level attention scores
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention extraction
            _, attention_dict = self.model(
                x, edge_index, batch, return_attention=True
            )
            
            # Get attention from both layers
            edge_index_1, attention_1 = attention_dict["layer1"]
            edge_index_2, attention_2 = attention_dict["layer2"]
            
            # Aggregate node-level attention
            num_nodes = x.size(0)
            node_attention = self._aggregate_node_attention(
                edge_index_1, attention_1,
                edge_index_2, attention_2,
                num_nodes
            )
            
            # Aggregate edge-level attention
            edge_attention = self._aggregate_edge_attention(
                edge_index, edge_index_1, attention_1,
                edge_index_2, attention_2
            )
            
            result = {
                'layer1_attention': attention_1,
                'layer2_attention': attention_2,
                'layer1_edge_index': edge_index_1,
                'layer2_edge_index': edge_index_2,
                'node_attention': node_attention,
                'edge_attention': edge_attention
            }
            
        return result
    
    def extract_gam_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract importance scores from GAM model using gradient-based attribution.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Dictionary containing:
                - node_importance: Node importance scores
                - feature_importance: Feature-level importance
        """
        self.model.eval()
        
        # Enable gradient computation
        x_grad = x.clone().requires_grad_(True)
        
        # Forward pass
        logits = self.model(x_grad, edge_index, batch)
        
        # Get malware class score
        malware_score = logits[:, 1].sum()
        
        # Backward pass
        malware_score.backward()
        
        # Node importance = gradient magnitude
        node_importance = torch.abs(x_grad.grad).sum(dim=1)
        
        # Feature importance = gradient per feature
        feature_importance = torch.abs(x_grad.grad).mean(dim=0)
        
        result = {
            'node_importance': node_importance.detach(),
            'feature_importance': feature_importance.detach()
        }
        
        return result
    
    def _aggregate_node_attention(
        self,
        edge_index_1: torch.Tensor,
        attention_1: torch.Tensor,
        edge_index_2: torch.Tensor,
        attention_2: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate attention scores to node level.
        
        Node attention = average attention received from all incoming edges.
        
        Args:
            edge_index_1: Edge index from layer 1
            attention_1: Attention scores from layer 1
            edge_index_2: Edge index from layer 2
            attention_2: Attention scores from layer 2
            num_nodes: Total number of nodes
            
        Returns:
            Node attention scores [num_nodes]
        """
        node_scores = torch.zeros(num_nodes, device=self.device)
        node_counts = torch.zeros(num_nodes, device=self.device)
        
        # Layer 1 attention
        for i in range(edge_index_1.size(1)):
            target_node = edge_index_1[1, i]
            # Average across attention heads
            attention_value = attention_1[i].mean()
            node_scores[target_node] += attention_value
            node_counts[target_node] += 1
        
        # Layer 2 attention (weighted more as it's deeper)
        for i in range(edge_index_2.size(1)):
            target_node = edge_index_2[1, i]
            attention_value = attention_2[i].mean()
            node_scores[target_node] += attention_value * 1.5  # Weight layer 2 more
            node_counts[target_node] += 1
        
        # Normalize by count
        node_counts = torch.clamp(node_counts, min=1)
        node_attention = node_scores / node_counts
        
        return node_attention
    
    def _aggregate_edge_attention(
        self,
        original_edge_index: torch.Tensor,
        edge_index_1: torch.Tensor,
        attention_1: torch.Tensor,
        edge_index_2: torch.Tensor,
        attention_2: torch.Tensor
    ) -> Dict[Tuple[int, int], float]:
        """
        Aggregate attention scores to edge level.
        
        Args:
            original_edge_index: Original graph edges
            edge_index_1: Edge index from layer 1
            attention_1: Attention from layer 1
            edge_index_2: Edge index from layer 2
            attention_2: Attention from layer 2
            
        Returns:
            Dictionary mapping edge (u, v) to attention score
        """
        edge_attention = {}
        
        # Layer 1
        for i in range(edge_index_1.size(1)):
            u = edge_index_1[0, i].item()
            v = edge_index_1[1, i].item()
            attention_value = attention_1[i].mean().item()
            
            edge_key = (u, v)
            if edge_key not in edge_attention:
                edge_attention[edge_key] = []
            edge_attention[edge_key].append(attention_value)
        
        # Layer 2
        for i in range(edge_index_2.size(1)):
            u = edge_index_2[0, i].item()
            v = edge_index_2[1, i].item()
            attention_value = attention_2[i].mean().item()
            
            edge_key = (u, v)
            if edge_key not in edge_attention:
                edge_attention[edge_key] = []
            edge_attention[edge_key].append(attention_value)
        
        # Average attention per edge
        edge_attention_avg = {
            edge: np.mean(scores) 
            for edge, scores in edge_attention.items()
        }
        
        return edge_attention_avg
    
    def extract_top_k_nodes(
        self,
        node_attention: torch.Tensor,
        node_names: List[str],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract top-k most important nodes.
        
        Args:
            node_attention: Node attention scores
            node_names: List of node names
            k: Number of top nodes to return
            
        Returns:
            List of (node_name, attention_score) tuples
        """
        if isinstance(node_attention, torch.Tensor):
            scores = node_attention.cpu().numpy()
        else:
            scores = np.array(node_attention)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        top_k_nodes = [
            (node_names[idx], float(scores[idx]))
            for idx in top_k_indices
            if idx < len(node_names)
        ]
        
        return top_k_nodes
    
    def extract_top_k_edges(
        self,
        edge_attention: Dict[Tuple[int, int], float],
        node_names: List[str],
        k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Extract top-k most important edges.
        
        Args:
            edge_attention: Edge attention scores
            node_names: List of node names
            k: Number of top edges to return
            
        Returns:
            List of (source_node, target_node, attention_score) tuples
        """
        # Sort edges by attention
        sorted_edges = sorted(
            edge_attention.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        top_k_edges = [
            (node_names[u], node_names[v], score)
            for (u, v), score in sorted_edges
            if u < len(node_names) and v < len(node_names)
        ]
        
        return top_k_edges
    
    def compare_attention(
        self,
        attention1: torch.Tensor,
        attention2: torch.Tensor,
        node_names: List[str]
    ) -> Dict:
        """
        Compare attention from two different models (e.g., GAT vs GAM).
        
        Args:
            attention1: Attention from first model
            attention2: Attention from second model
            node_names: List of node names
            
        Returns:
            Dictionary with comparison statistics
        """
        if isinstance(attention1, torch.Tensor):
            att1 = attention1.cpu().numpy()
        else:
            att1 = np.array(attention1)
        
        if isinstance(attention2, torch.Tensor):
            att2 = attention2.cpu().numpy()
        else:
            att2 = np.array(attention2)
        
        # Normalize for fair comparison
        att1_norm = att1 / (np.max(att1) + 1e-8)
        att2_norm = att2 / (np.max(att2) + 1e-8)
        
        # Compute correlation
        correlation = np.corrcoef(att1_norm, att2_norm)[0, 1]
        
        # Find nodes with high agreement
        diff = np.abs(att1_norm - att2_norm)
        agreement_indices = np.where(diff < 0.1)[0]
        
        # Find nodes with disagreement
        disagreement_indices = np.where(diff > 0.5)[0]
        
        comparison = {
            'correlation': float(correlation),
            'mean_difference': float(np.mean(diff)),
            'max_difference': float(np.max(diff)),
            'agreement_ratio': len(agreement_indices) / len(att1),
            'high_agreement_nodes': [
                node_names[idx] for idx in agreement_indices[:10]
                if idx < len(node_names)
            ],
            'high_disagreement_nodes': [
                (node_names[idx], float(att1[idx]), float(att2[idx]))
                for idx in disagreement_indices[:10]
                if idx < len(node_names)
            ]
        }
        
        return comparison
    
    def get_attention_statistics(
        self,
        attention: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get statistical summary of attention scores.
        
        Args:
            attention: Attention scores
            
        Returns:
            Dictionary of statistics
        """
        if isinstance(attention, torch.Tensor):
            scores = attention.cpu().numpy()
        else:
            scores = np.array(attention)
        
        stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'non_zero_ratio': float(np.sum(scores > 0) / len(scores))
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with dummy data
    from src.models.gat_model import GAT
    
    model = GAT(num_node_features=15, hidden_units=8, num_heads=8)
    extractor = AttentionExtractor(model, device="cpu")
    
    # Create dummy graph
    num_nodes = 50
    num_edges = 100
    
    x = torch.randn(num_nodes, 15)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Extract attention
    attention = extractor.extract_gat_attention(x, edge_index, batch)
    
    print("Attention Extraction Results:")
    print(f"  Node attention shape: {attention['node_attention'].shape}")
    print(f"  Node attention stats: {extractor.get_attention_statistics(attention['node_attention'])}")
    
    # Create node names
    node_names = [f"method_{i}" for i in range(num_nodes)]
    
    # Get top-k nodes
    top_nodes = extractor.extract_top_k_nodes(
        attention['node_attention'], node_names, k=5
    )
    
    print("\nTop 5 Important Nodes:")
    for node, score in top_nodes:
        print(f"  {node}: {score:.6f}")