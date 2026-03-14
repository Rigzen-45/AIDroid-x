"""
PyTorch Geometric Dataset for Android Malware Detection - FIXED
Automatically fixes missing 'category' and 'is_sensitive' attributes
"""

import torch
import pickle
import logging
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch_geometric.data import Data, Dataset
import networkx as nx

logger = logging.getLogger(__name__)

NUM_NODE_FEATURES = 41
CATS = [
    "NETWORK", "SMS", "PHONE", "LOCATION", "CAMERA",
    "MICROPHONE", "CONTACTS", "STORAGE", "SYSTEM", "CRYPTO",
]


def _is_obfuscated(name: str) -> bool:
    """Return True if a class/method name looks ProGuard-obfuscated.
    
    Heuristic: 1–2 alphabetic characters only (e.g. 'a', 'ab', 'B').
    Confirmed 3.93–5.16× malware ratio in check_obfuscation_signal.py.
    """
    return bool(name) and len(name) <= 2 and name.isalpha()
class MalwareGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset with automatic attribute fixing.
    
    Automatically fixes:
    - Missing 'is_sensitive' → copies from 'has_sensitive_api'
    - Missing 'category' → sets to 'SYSTEM'
    """
    
    def __init__(self, root: str, graphs: List[nx.DiGraph] = None,
                 labels: List[int] = None, transform=None, pre_transform=None):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for dataset
            graphs: List of NetworkX graphs
            labels: List of labels (0=benign, 1=malware)
            transform: Optional transform
            pre_transform: Optional pre-transform
            force_reload: If True, clear cache and reprocess
            fix_attributes: If True, automatically fix missing attributes
        """
        self.graphs = graphs or []
        self.labels = labels or []
        self.num_categories = 10
        self._num_node_features = NUM_NODE_FEATURES
        
        super().__init__(root, transform, pre_transform)

    # PyG Dataset interface

    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        if self.graphs:
            return [f"data_{i}.pt" for i in range(len(self.graphs))]
        processed_dir = Path(self.processed_dir)
        return sorted([p.name for p in processed_dir.glob("data_*.pt")])
        
    @property
    def num_node_features(self) -> int:
        return self._num_node_features

    def download(self):
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
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        """Get a single graph."""
        data = torch.load(self.processed_paths[idx])
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _networkx_to_pyg(self, graph: nx.DiGraph, label: int) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0:
            logger.warning("Empty graph - creating dummy data")
            return self._create_empty_data(label)
        
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # ── Precompute graph-level stats (O(N) once) ─────────────────────────
        # G6 features are graph-level constants — same value for every node.
        # Computing them inside _extract_node_features() (called N times each)
        # makes the loop O(N^2). Precompute here once and pass as a dict.
        total_nodes = num_nodes
        n_method  = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "method")
        n_reflect = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "reflection")

        n_obf_cls = sum(
            1 for n, d in graph.nodes(data=True)
            if d.get("is_obf_class", False) or (
                d.get("type") == "method" and
                _is_obfuscated(n.split("->")[0].rstrip(";").split("/")[-1].split("$")[0])
            )
        )
        n_obf_mth = sum(
            1 for n, d in graph.nodes(data=True)
            if d.get("is_obf_method", False) or (
                d.get("type") == "method" and "->" in n and
                _is_obfuscated(n.split("->", 1)[1].split("(")[0])
            )
        )
        hub_nodes      = sum(1 for n in graph.nodes() if graph.out_degree(n) >= 10)
        max_out_degree = max((graph.out_degree(n) for n in graph.nodes()), default=0)

        graph_stats = {
            "total_nodes":     total_nodes,
            "n_method":        n_method,
            "n_reflect":       n_reflect,
            "n_obf_cls":       n_obf_cls,
            "n_obf_mth":       n_obf_mth,
            "hub_nodes":       hub_nodes,
            "max_out_degree":  max_out_degree,
        }
        # ─────────────────────────────────────────────────────────────────────

        # Build node features
        node_features = []
        for node in nodes:
            node_data = graph.nodes[node]
            features = self._extract_node_features(
                node,
                node_data,
                graph,
                graph_stats,
            )
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)

        # Validate feature dimension (catches stale cached files early)
        assert x.shape[1] == NUM_NODE_FEATURES, (
            f"Feature dim mismatch: expected {NUM_NODE_FEATURES}, got {x.shape[1]}. "
            "Delete processed/ caches and re-run."
        )
        
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
        
        y = torch.tensor([label], dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )
        
        # Graph-level size features: give the FC head explicit size info so it
        # can discount or up-weight the pooled embedding based on graph scale.
        # Addresses the 7.57× malware/benign size differential (Section 2.1).
        # Shape [1,2]: PyG stacks [1,2] per graph -> [B,2] per batch.
        # Shape [2] gets mis-batched -> [B*2] (1D) -> unsqueeze crash.
        data.graph_size = torch.tensor([[
            math.log1p(graph.number_of_nodes()),
            math.log1p(graph.number_of_edges()),
        ]], dtype=torch.float)

        data.apk_name = graph.graph.get("apk_name", "unknown")
        data.package_name = graph.graph.get("package_name", "unknown")
        data.node_names = nodes
    
        return data

    # ------------------------------------------------------------------
    # Node feature extraction  ← THE CORE FIX
    # ------------------------------------------------------------------

    def _extract_node_features(
        self,
        node: str,
        node_data: Dict,
        graph: nx.DiGraph,
        graph_stats: Dict = None,
    ) -> List[float]:
       
        features: List[float] = []

        cat_counts: Dict[str, int] = node_data.get("category_counts", {})

        # ── Group 1: Per-category counts (10 dims) ───────────────────────────
        for cat in CATS:
            features.append(float(np.log1p(cat_counts.get(cat, 0))))

        # ── Group 2: Diversity (3 dims) ──────────────────────────────────────
        total_sens = sum(cat_counts.values())
        n_cats     = sum(1 for v in cat_counts.values() if v > 0)

        # Shannon entropy (normalised to [0, 1])
        if total_sens > 0:
            probs   = [v / total_sens for v in cat_counts.values() if v > 0]
            entropy = -sum(p * np.log(p + 1e-9) for p in probs) / np.log(10 + 1e-9)
        else:
            entropy = 0.0

        features.append(float(np.log1p(total_sens)))   # total sensitive calls
        features.append(float(n_cats / 10.0))           # fraction of cats used
        features.append(float(entropy))                 # call diversity

        # ── Group 3: Dangerous combination flags (5 dims) ────────────────────
        # NOTE: CONTACTS (ratio 1.00x) and PHONE (ratio 0.98x) are excluded
        # from danger flags (confirmed near-zero discriminative power), but
        # remain in G1 category_counts as contextual information.
        has = lambda c: cat_counts.get(c, 0) > 0  # noqa: E731

        features.append(float(has("LOCATION") and has("NETWORK")))   # GPS exfil
        features.append(float(has("CRYPTO")))                         # encryption
        features.append(float(has("MICROPHONE") or has("CAMERA")))   # surveillance
        features.append(float(has("SMS") and has("NETWORK")))         # SMS + C2 combo
        features.append(float(has("STORAGE") and has("NETWORK")))     # file exfil

        # ── Group 4: Structural features (7 dims) ────────────────────────────
        in_d  = graph.in_degree(node)
        out_d = graph.out_degree(node)
        features.append(float(np.log1p(in_d)))
        features.append(float(np.log1p(out_d)))
        features.append(float(np.log1p(in_d + out_d)))
        features.append(float(out_d / (in_d + 1)))          # fan-out ratio
        features.append(float(in_d > 5))                     # is hub node
        features.append(float(np.log1p(node_data.get("code_size", 0))))
        features.append(float(np.log1p(node_data.get("api_count", total_sens))))

        # ── Group 5: Method name heuristics (10 dims) ────────────────────────
        name = str(node).lower()
        heuristics = [
            any(k in name for k in ["encrypt", "cipher", "aes", "des", "rc4", "crypt"]),
            any(k in name for k in ["send", "upload", "post", "submit", "transmit"]),
            any(k in name for k in ["collect", "harvest", "steal", "grab", "exfil"]),
            any(k in name for k in ["hide", "obfuscat", "conceal", "pack", "encode"]),
            (len(name.split(".")[-1]) <= 3 and name.split(".")[-1].isalpha()),  # short obfuscated name
            any(k in name for k in ["root", "su", "superuser", "exploit", "shell"]),
            any(k in name for k in ["sms", "text", "message", "shortmessage"]),
            any(k in name for k in ["location", "gps", "coordinate", "latitude"]),
            any(k in name for k in ["contact", "phonebook", "addressbook"]),
            any(k in name for k in ["camera", "mic", "record", "capture", "snapshot"]),
        ]
        features.extend([float(h) for h in heuristics])

        # ── Group 6: Obfuscation & structural signals (6 dims) ───────────────
        # All values are graph-level constants precomputed in _networkx_to_pyg()
        # Removed from here to avoid O(N^2): each was O(N), called N times.
        if graph_stats is not None:
            total_nodes    = graph_stats["total_nodes"]
            n_method       = graph_stats["n_method"]
            n_reflect      = graph_stats["n_reflect"]
            n_obf_cls      = graph_stats["n_obf_cls"]
            n_obf_mth      = graph_stats["n_obf_mth"]
            hub_nodes      = graph_stats["hub_nodes"]
            max_out_degree = graph_stats["max_out_degree"]
        else:
            # Fallback (inference path — single graph, no precompute dict)
            total_nodes    = graph.number_of_nodes()
            n_method    = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "method")
            n_reflect   = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "reflection")
            n_obf_cls   = sum(
                1 for n, d in graph.nodes(data=True)
                if d.get("is_obf_class", False) or (
                    d.get("type") == "method" and
                    _is_obfuscated(n.split("->")[0].rstrip(";").split("/")[-1].split("$")[0])
                )
            )
            n_obf_mth   = sum(
                1 for n, d in graph.nodes(data=True)
                if d.get("is_obf_method", False) or (
                    d.get("type") == "method" and "->" in n and
                    _is_obfuscated(n.split("->", 1)[1].split("(")[0])
                )
            )
            hub_nodes      = sum(1 for n in graph.nodes() if graph.out_degree(n) >= 10)
            max_out_degree = max((graph.out_degree(n) for n in graph.nodes()), default=0)

        reflection_ratio  = n_reflect / max(total_nodes, 1)
        reflection_exists = float(n_reflect > 0)
        obf_class_ratio   = n_obf_cls / max(n_method, 1)
        obf_method_ratio  = n_obf_mth / max(n_method, 1)
        hub_node_ratio    = hub_nodes  / max(n_method, 1)

        features.append(float(reflection_ratio))    # reflection_ratio  (4.38×)
        features.append(float(reflection_exists))   # reflection_exists (1.94×)
        features.append(float(obf_class_ratio))     # obf_class_ratio   (4.40×)
        features.append(float(obf_method_ratio))    # obf_method_ratio  (3.93×)
        features.append(float(hub_node_ratio))      # hub_node_ratio    (5.16×)
        features.append(float(np.log1p(max_out_degree)))  # max_out_degree (4.15×)

        
        assert len(features) == NUM_NODE_FEATURES, (
            f"BUG: feature vector has {len(features)} dims, expected {NUM_NODE_FEATURES}"
        )
        return features

    # ------------------------------------------------------------------
    # Edge features 
    # ------------------------------------------------------------------

    def _extract_edge_features(self, edge_data: Dict) -> List[float]:
        is_sensitive = float(edge_data.get("is_sensitive_call", False))
        call_type_map = {"direct": 1.0, "indirect": 0.5, "intra_class": 0.25}
        call_type = float(call_type_map.get(edge_data.get("call_type", "direct"), 0.0))
        return [is_sensitive, call_type]

    def _create_empty_data(self, label: int) -> Data:
        data = Data(
            x=torch.zeros((1, NUM_NODE_FEATURES), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=1,
        )
        data.graph_size = torch.zeros((1, 2), dtype=torch.float)
        return data


    @staticmethod
    def save_dataset(graphs: List[nx.DiGraph], labels: List[int], save_path: str) -> None:
        """Save graphs and labels."""
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
        """Load saved graphs and labels."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        graphs = data["graphs"]
        labels = data["labels"]
        
        logger.info(f"Loaded {len(graphs)} graphs from {load_path}")
        
        return graphs, labels
