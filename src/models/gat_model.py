"""
Graph Attention Network (GAT) for Malware Detection
FIXED VERSION - Now supports class-weighted loss
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
    - Input layer: Node features (NUM_NODE_FEATURES=41)
    - GAT Layer 1: num_heads × hidden_units with attention
    - GAT Layer 2: num_heads × hidden_units with attention
    - Global pooling: Mean + Max concatenation
    - Graph size features: log1p(num_nodes), log1p(num_edges) appended to pool
    - FC layers: Classification head
    
    Graph size features address the 7.57× malware/benign node-count
    differential that causes global mean pooling to spuriously encode
    graph scale as a malware signal (false positives on large benign apps).
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_units: int = 32,       # 32 units/head (was 8/16) — doubled capacity
        num_heads: int = 8,           # 8 heads (was 4) — more diverse attention
        num_classes: int = 2,
        dropout: float = 0.3,
        attention_dropout: float = 0.4,  # slightly reduced for larger dataset
        use_graph_size: bool = True,     # concat log1p(nodes,edges) before FC head
    ):
        """
        Initialize GAT model.
        
        Args:
            num_node_features: Dimension of input node features (41 with obf features)
            hidden_units: Hidden units per attention head (32 recommended for 10K dataset)
            num_heads: Number of attention heads (8 recommended for 10K dataset)
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability for FC layers (0.3 — less aggressive than 0.5)
            attention_dropout: Dropout for attention coefficients (0.4)
            use_graph_size: If True, appends 2 graph-size scalars to pooled embedding
        """
        super(GAT, self).__init__()
        
        self.num_node_features = num_node_features
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_graph_size = use_graph_size
        
        # Store attention scores for explainability
        self.attention_weights_layer1 = None
        self.attention_weights_layer2 = None
        
        # GAT Layer 1: (num_features) -> (num_heads * hidden_units)
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_units,
            heads=num_heads,
            dropout=attention_dropout,
            concat=True,
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
        # e.g. 8 heads × 32 units × 2 (mean+max) = 512d
        pool_dim = hidden_units * num_heads * 2
        
        # +2 for graph_size features (log1p num_nodes, log1p num_edges)
        fc_input_dim = pool_dim + (2 if use_graph_size else 0)
        
        # Classification head — dropout reduced from 0.5 to 0.3 (FC head is
        # small at ~8K params; aggressive dropout slows convergence needlessly)
        self.fc1 = nn.Linear(fc_input_dim, 64)
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
        return_attention: bool = False,
        graph_size: Optional[torch.Tensor] = None,  # [batch_size, 2] size features
    ) -> Tuple[torch.Tensor, Optional[Dict]]:

        attention1 = None
        attention2 = None

        # -------------------------
        # GAT Layer 1
        # -------------------------
        if return_attention:
            x, attention1 = self.conv1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv1(x, edge_index)

        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # -------------------------
        # GAT Layer 2
        # -------------------------
        if return_attention:
            x, attention2 = self.conv2(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv2(x, edge_index)

        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # -------------------------
        # Global Pooling
        # -------------------------
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  # pool_dim (e.g. 512d)

        # -------------------------
        # Graph Size Features
        # Concatenate log1p(num_nodes), log1p(num_edges) so the FC head
        # can explicitly condition on graph scale — preventing the pooling
        # layer from using size as a spurious malware proxy.
        # -------------------------
        if self.use_graph_size and graph_size is not None:
            # view(-1, 2) handles any shape PyG produces:
            #   [2]      (unbatched)   -> [1,  2]
            #   [B*2]    (cat-batched) -> [B,  2]
            #   [B, 2]   (stack-batch) -> [B,  2]  (no-op)
            x = torch.cat([x, graph_size.view(-1, 2).to(x.device)], dim=1)

        # -------------------------
        # Classification Head
        # dropout=0.3 (reduced from 0.5 — FC head is small ~8K params,
        # 0.5 was too aggressive and slowed convergence on small datasets)
        # -------------------------
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        logits = self.fc3(x)

        # -------------------------
        # Attention Output
        # -------------------------
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

            # Unpack attention
            edge_index_1, attention_1 = attention_dict["layer1"]
            edge_index_2, attention_2 = attention_dict["layer2"]

            num_nodes = x.size(0)
            device = x.device

            node_scores = torch.zeros(num_nodes, device=device)

            # ---------
            # Layer 1
            # ---------
            target_nodes_1 = edge_index_1[1]                 # [num_edges]
            attn_values_1 = attention_1.mean(dim=1)          # avg over heads

            node_scores.scatter_add_(0, target_nodes_1, attn_values_1)

            # ---------
            # Layer 2
            # ---------
            target_nodes_2 = edge_index_2[1]
            attn_values_2 = attention_2.mean(dim=1)

            node_scores.scatter_add_(0, target_nodes_2, attn_values_2)

            
            # Correct in-degree (only once!)
            in_degrees = torch.bincount(
                edge_index_1[1],
                minlength=num_nodes
            ).float().to(device)

            in_degrees = torch.clamp(in_degrees, min=1.0)

            node_scores = node_scores / in_degrees
            
        return node_scores
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            graph_size: Optional [batch_size, 2] graph size features
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(
                x, edge_index, batch,
                return_attention=False,
                graph_size=graph_size,
            )
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class GATTrainer:
    """Trainer class for GAT model with class-weighted loss support."""
    
    def __init__(
        self,
        model: GAT,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        criterion = None,
        class_weights: Optional[torch.Tensor] = None  # NEW: Support class weights
    ):
        """
        Initialize trainer.
        
        Args:
            model: GAT model
            optimizer: PyTorch optimizer
            device: Device to train on
            criterion: Loss function (if None, creates CrossEntropyLoss with class_weights)
            class_weights: Tensor of class weights [num_classes] (IMPORTANT!)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # CRITICAL FIX: Use class weights if provided
        if criterion is not None:
            self.criterion = criterion
        elif class_weights is not None:
            # Use class-weighted loss
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            logger.info(f"Using class-weighted CrossEntropyLoss with weights: {class_weights.tolist()}")
        else:
            # Fallback to unweighted loss (not recommended!)
            self.criterion = nn.CrossEntropyLoss()
            logger.warning("No class weights provided - using unweighted loss (may cause imbalance issues)")
        
        logger.info(f"Initialized GAT trainer on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            
            # Extract graph_size if present (added by graph_dataset.py to address
            # the 7.57× malware/benign size differential — Section 2.1 of report)
            graph_size = getattr(batch, "graph_size", None)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(
                batch.x, batch.edge_index, batch.batch,
                return_attention=False,
                graph_size=graph_size,
            )
            
            # Compute loss
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping — prevents instability on high-degree nodes
            # (max_out_degree up to 15.8 in malware graphs produces large grads)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                
                graph_size = getattr(batch, "graph_size", None)
                
                logits, _ = self.model(
                    batch.x, batch.edge_index, batch.batch,
                    return_attention=False,
                    graph_size=graph_size,
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
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        return avg_loss, accuracy, f1
