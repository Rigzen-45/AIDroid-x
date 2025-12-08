"""
PyTorch Geometric Dataset for Android Malware Detection
Converts NetworkX graphs to PyG Data objects with proper features
"""

import torch
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx

logger = logging.getLogger(__name__)


class MalwareGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for malware/benign APK graphs.
    
    Features per node:
    - One-hot encoded API category (10 categories)
    - Is sensitive API (binary)
    - In-degree, out-degree
    - Code size (for method nodes)
    - API call frequency
    """
    
    def __init__(self, root: str, graphs: List[nx.DiGraph] = None,
                 labels: List[int] = None, transform=None, pre_transform=None):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for dataset
            graphs: List of NetworkX graphs (if None, load from processed)
            labels: List of labels (0=benign, 1=malware)
            transform: Optional transform to apply to each sample
            pre_transform: Optional pre-transform before caching
        """
        self.graphs = graphs or []
        self.labels = labels or []
        self.num_categories = 10  # API categories from config
        
        # Feature dimensions
        self._num_node_features = self.num_categories + 5  # category + 5 numeric
        
        # Label encoder for categories
        self.category_encoder = LabelEncoder()
        self._fit_category_encoder()
        
        super().__init__(root, transform, pre_transform)
    
    def _fit_category_encoder(self):
        """Fit label encoder with all possible categories."""
        all_categories = [
            "NETWORK", "SMS", "PHONE", "LOCATION", "CAMERA",
            "MICROPHONE", "CONTACTS", "STORAGE", "SYSTEM", "CRYPTO"
        ]
        self.category_encoder.fit(all_categories)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files (NetworkX pickles)."""
        return []  # We handle this manually
    
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files (PyG Data objects)."""
        return [f"data_{i}.pt" for i in range(len(self.graphs))]
    
    @property
    def num_node_features(self):
        return self._num_node_features

    def download(self):
        """Download dataset (not needed for our case)."""
        pass
    
    def process(self):
        """
        Process raw NetworkX graphs into PyG Data objects.
        Called automatically if processed files don't exist.
        """
        logger.info(f"Processing {len(self.graphs)} graphs...")
        
        for idx, (graph, label) in enumerate(zip(self.graphs, self.labels)):
            # Convert NetworkX graph to PyG Data
            data = self._networkx_to_pyg(graph, label)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # Save processed data
            torch.save(data, self.processed_paths[idx])
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.graphs)} graphs")
    
    def len(self) -> int:
        """Return dataset size."""
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        """
        Get a single graph.
        
        Args:
            idx: Index
            
        Returns:
            PyG Data object
        """
        data = torch.load(self.processed_paths[idx])
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _networkx_to_pyg(self, graph: nx.DiGraph, label: int) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            graph: NetworkX DiGraph
            label: Graph label (0 or 1)
            
        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features]
                - y: Graph label [1]
        """
        # Get node list (fixed order)
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0:
            logger.warning("Empty graph encountered")
            return self._create_empty_data(label)
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Build node features
        node_features = []
        for node in nodes:
            node_data = graph.nodes[node]
            features = self._extract_node_features(node, node_data, graph)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Build edge index
        edge_list = []
        edge_features = []
        for u, v, edge_data in graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                edge_features.append(self._extract_edge_features(edge_data))
        
        if not edge_list:
            logger.warning(f"No edges in graph with {num_nodes} nodes")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Graph label
        y = torch.tensor([label], dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )
        
        # Store metadata
        data.apk_name = graph.graph.get("apk_name", "unknown")
        data.package_name = graph.graph.get("package_name", "unknown")
        data.node_names = nodes  # For later code localization
        
        return data
    
    def _extract_node_features(self, node: str, node_data: Dict, 
                              graph: nx.DiGraph) -> List[float]:
        """
        Extract feature vector for a node.
        
        Features (15 total):
        - One-hot encoded category (10 dimensions)
        - Is sensitive (1)
        - In-degree (1)
        - Out-degree (1)
        - Code size (1)
        - API call count (1)
        
        Args:
            node: Node ID
            node_data: Node attributes
            graph: Full graph for degree calculation
            
        Returns:
            Feature vector as list
        """
        features = []
        
        # One-hot encode category (10 dimensions)
        category = node_data.get("category", "UNKNOWN")
        if category == "UNKNOWN" or category not in self.category_encoder.classes_:
            # Default to zeros if unknown
            category_onehot = [0.0] * self.num_categories
        else:
            category_id = self.category_encoder.transform([category])[0]
            category_onehot = [0.0] * self.num_categories
            category_onehot[category_id] = 1.0
        features.extend(category_onehot)
        
        # Is sensitive API
        is_sensitive = float(node_data.get("is_sensitive", False) or 
                           node_data.get("has_sensitive_api", False))
        features.append(is_sensitive)
        
        # Degree features (normalized by log)
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        features.append(np.log1p(in_degree))  # log(1 + x) for stability
        features.append(np.log1p(out_degree))
        
        # Code size (for methods)
        code_size = node_data.get("code_size", 0)
        features.append(np.log1p(code_size))
        
        # API call count
        api_count = node_data.get("api_count", 0)
        features.append(np.log1p(api_count))
        
        return features
    
    def _extract_edge_features(self, edge_data: Dict) -> List[float]:
        """
        Extract edge features.
        
        Features (2 total):
        - Is sensitive call (1)
        - Call type encoded (1): direct=1, indirect=0.5, intra_class=0.25
        
        Args:
            edge_data: Edge attributes
            
        Returns:
            Feature vector as list
        """
        features = []
        
        # Is sensitive call
        is_sensitive = float(edge_data.get("is_sensitive_call", False))
        features.append(is_sensitive)
        
        # Call type encoding
        call_type = edge_data.get("call_type", "direct")
        call_type_map = {
            "direct": 1.0,
            "indirect": 0.5,
            "intra_class": 0.25
        }
        features.append(call_type_map.get(call_type, 0.0))
        
        return features
    
    def _create_empty_data(self, label: int) -> Data:
        """
        Create empty Data object for edge cases.
        
        Args:
            label: Graph label
            
        Returns:
            Empty PyG Data object
        """
        return Data(
            x=torch.zeros((1, self.num_node_features), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=1
        )
    
    @staticmethod
    def save_dataset(graphs: List[nx.DiGraph], labels: List[int], 
                    save_path: str) -> None:
        """
        Save graphs and labels for later loading.
        
        Args:
            graphs: List of NetworkX graphs
            labels: List of labels
            save_path: Path to save pickle file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "graphs": graphs,
            "labels": labels
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(graphs)} graphs to {save_path}")
    
    @staticmethod
    def load_dataset(load_path: str) -> Tuple[List[nx.DiGraph], List[int]]:
        """
        Load saved graphs and labels.
        
        Args:
            load_path: Path to pickle file
            
        Returns:
            Tuple of (graphs, labels)
        """
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        graphs = data["graphs"]
        labels = data["labels"]
        
        logger.info(f"Loaded {len(graphs)} graphs from {load_path}")
        
        return graphs, labels


# Example usage and dataset creation
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dummy data for testing
    def create_dummy_graph(num_nodes: int = 20) -> nx.DiGraph:
        """Create a dummy graph for testing."""
        G = nx.DiGraph()
        
        for i in range(num_nodes):
            G.add_node(
                f"method_{i}",
                type="method",
                category="NETWORK" if i % 3 == 0 else "PHONE",
                is_sensitive=i % 4 == 0,
                code_size=np.random.randint(10, 100),
                api_count=np.random.randint(1, 10)
            )
        
        for i in range(num_nodes - 1):
            if np.random.random() > 0.5:
                G.add_edge(
                    f"method_{i}",
                    f"method_{i+1}",
                    call_type="direct",
                    is_sensitive_call=i % 5 == 0
                )
        
        G.graph["apk_name"] = "test.apk"
        G.graph["package_name"] = "com.test.app"
        
        return G
    
    # Create test dataset
    graphs = [create_dummy_graph() for _ in range(10)]
    labels = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # Mix of benign/malware
    
    # Initialize dataset
    dataset = MalwareGraphDataset(
        root="data/test_dataset",
        graphs=graphs,
        labels=labels
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Node features: {dataset.num_node_features}")
    
    # Test loading
    sample = dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Label: {sample.y.item()}")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge features shape: {sample.edge_attr.shape}")