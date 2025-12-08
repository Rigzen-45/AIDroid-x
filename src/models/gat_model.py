"""
Graph Attention Network (GAT) for Malware Detection
With attention score extraction for explainability
Paper params: 8 heads, 8 hidden units, LR=0.0005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GAT(nn.Module):
    """
    Graph Attention Network for binary malware classification.
    
    Architecture:
    - Input layer: Node features
    - GAT Layer 1: num_heads × hidden_units with attention
    - GAT Layer 2: num_heads × hidden_units with attention
    - Global pooling: Mean + Max concatenation
    - FC layers: Classification head
    
    Attention scores are stored for explainability.
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_units: int = 8,
        num_heads: int = 8,
        num_classes: int = 2,
        dropout: float = 0.6,
        attention_dropout: float = 0.6
    ):
        """
        Initialize GAT model.
        
        Args:
            num_node_features: Dimension of input node features
            hidden_units: Hidden units per attention head
            num_heads: Number of attention heads
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability for features
            attention_dropout: Dropout for attention coefficients
        """
        super(GAT, self).__init__()
        
        self.num_node_features = num_node_features
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Store attention scores for explainability
        self.attention_weights_layer1 = None
        self.attention_weights_layer2 = None
        
        # GAT Layer 1: (num_features) -> (num_heads * hidden_units)
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_units,
            heads=num_heads,
            dropout=attention_dropout,
            concat=True,  # Concatenate head outputs
            add_self_loops=True
        )
        
        # GAT Layer 2: (num_heads * hidden_units) -> (num_heads * hidden_units)
        self.conv2 = GATConv(
            in_channels=hidden_units * num_heads,
            out_channels=hidden_units,
            heads=num_heads,
            dropout=attention_dropout,
            concat=True,
            add_self_loops=True
        )
        
        # Graph-level representation: concat of mean and max pooling
        graph_embedding_size = hidden_units * num_heads * 2
        
        # Classification head
        self.fc1 = nn.Linear(graph_embedding_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_units * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_units * num_heads)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.bn_fc2 = nn.BatchNorm1d(32)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through GAT.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            return_attention: If True, return attention weights
            
        Returns:
            Tuple of:
                - logits: Classification logits [batch_size, num_classes]
                - attention_dict: Dictionary with attention weights (if requested)
        """
        # Store original number of nodes for attention extraction
        num_nodes = x.size(0)
        
        # GAT Layer 1
        x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store attention for explainability
        if return_attention or not self.training:
            self.attention_weights_layer1 = attention1
        
        # GAT Layer 2
        x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store attention for explainability
        if return_attention or not self.training:
            self.attention_weights_layer2 = attention2
        
        # Global pooling: Combine mean and max
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden * heads]
        x_max = global_max_pool(x, batch)    # [batch_size, hidden * heads]
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden * heads * 2]
        
        # Classification head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.fc3(x)
        
        # Prepare attention output
        attention_dict = None
        if return_attention:
            attention_dict = {
                "layer1": attention1,
                "layer2": attention2
            }
        
        return logits, attention_dict
    
    def extract_node_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention scores for each node (aggregate across heads and layers).
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            Node attention scores [num_nodes] (higher = more important)
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with attention extraction
            _, attention_dict = self.forward(
                x, edge_index, batch, return_attention=True
            )
            
            # Get attention from both layers
            edge_index_1, attention_1 = attention_dict["layer1"]
            edge_index_2, attention_2 = attention_dict["layer2"]
            
            # Aggregate attention scores per node
            num_nodes = x.size(0)
            node_scores = torch.zeros(num_nodes, device=x.device)
            
            # Layer 1: Average attention received by each node
            for i in range(edge_index_1.size(1)):
                target_node = edge_index_1[1, i].item()
                # Average across attention heads
                attention_value = attention_1[i].mean().item()
                node_scores[target_node] += attention_value
            
            # Layer 2: Add attention from second layer
            for i in range(edge_index_2.size(1)):
                target_node = edge_index_2[1, i].item()
                attention_value = attention_2[i].mean().item()
                node_scores[target_node] += attention_value
            
            # Normalize by number of incoming edges
            in_degrees = torch.zeros(num_nodes, device=x.device)
            for i in range(edge_index_1.size(1)):
                in_degrees[edge_index_1[1, i]] += 1
            for i in range(edge_index_2.size(1)):
                in_degrees[edge_index_2[1, i]] += 1
            
            # Avoid division by zero
            in_degrees = torch.clamp(in_degrees, min=1)
            node_scores = node_scores / in_degrees
            
        return node_scores
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index, batch, return_attention=False)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class GATTrainer:
    """Trainer class for GAT model."""
    
    def __init__(
        self,
        model: GAT,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        criterion = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: GAT model
            optimizer: PyTorch optimizer
            device: Device to train on
            criterion: Loss function (default: CrossEntropyLoss)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        logger.info(f"Initialized GAT trainer on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: PyG DataLoader
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(
                batch.x, batch.edge_index, batch.batch, return_attention=False
            )
            
            # Compute loss
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate(self, val_loader) -> Tuple[float, float, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: PyG DataLoader
            
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                logits, _ = self.model(
                    batch.x, batch.edge_index, batch.batch, return_attention=False
                )
                
                loss = self.criterion(logits, batch.y)
                total_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        return avg_loss, accuracy, f1


# Example usage
if __name__ == "__main__":
    # Test GAT model
    model = GAT(
        num_node_features=15,  # From dataset
        hidden_units=8,
        num_heads=8,
        num_classes=2,
        dropout=0.6
    )
    
    print(f"GAT Model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Architecture: {model}")
    
    # Test forward pass with dummy data
    batch_size = 2
    num_nodes = 50
    num_edges = 100
    
    x = torch.randn(num_nodes, 15)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.cat([torch.zeros(25, dtype=torch.long),
                      torch.ones(25, dtype=torch.long)])
    
    logits, attention = model(x, edge_index, batch, return_attention=True)
    print(f"\nTest forward pass:")
    print(f"  Input: {num_nodes} nodes, {num_edges} edges")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Attention layer 1 edges: {attention['layer1'][0].shape}")
    print(f"  Attention layer 2 edges: {attention['layer2'][0].shape}")