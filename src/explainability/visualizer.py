"""
Visualization Module
Creates attention heatmaps, graph visualizations, and analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualizes attention scores from GAT/GAM models.
    Creates heatmaps, distribution plots, and attention overlays.
    """
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        node_names: List[str],
        title: str = "Attention Heatmap",
        save_name: str = "attention_heatmap.png",
        top_k: int = 30
    ) -> str:
        """
        Create heatmap of attention scores.
        
        Args:
            attention_scores: Attention scores [num_nodes]
            node_names: Node names
            title: Plot title
            save_name: Filename to save
            top_k: Show only top-k nodes
            
        Returns:
            Path to saved figure
        """
        # Get top-k nodes
        top_k_indices = np.argsort(attention_scores)[-top_k:][::-1]
        top_scores = attention_scores[top_k_indices]
        top_names = [node_names[i] for i in top_k_indices]
        
        # Shorten method names for display
        display_names = []
        for name in top_names:
            if "->" in name:
                class_name = name.split("->")[0].split("/")[-1].replace(";", "")
                method_name = name.split("->")[1].split("(")[0]
                display_names.append(f"{class_name}.{method_name}")
            else:
                display_names.append(name[-30:])  # Last 30 chars
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.3)))
        
        # Reshape for heatmap
        scores_2d = top_scores.reshape(-1, 1)
        
        sns.heatmap(
            scores_2d,
            annot=True,
            fmt='.6f',
            cmap='YlOrRd',
            yticklabels=display_names,
            xticklabels=['Attention'],
            cbar_kws={'label': 'Attention Score'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention heatmap to {save_path}")
        return str(save_path)
    
    def plot_attention_distribution(
        self,
        attention_scores: np.ndarray,
        title: str = "Attention Distribution",
        save_name: str = "attention_distribution.png"
    ) -> str:
        """
        Plot distribution of attention scores.
        
        Args:
            attention_scores: Attention scores
            title: Plot title
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(attention_scores, bins=50, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Attention Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Histogram', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(attention_scores, vert=True)
        axes[1].set_ylabel('Attention Score', fontsize=12)
        axes[1].set_title('Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention distribution to {save_path}")
        return str(save_path)
    
    def plot_attention_comparison(
        self,
        gat_attention: np.ndarray,
        gam_attention: np.ndarray,
        node_names: List[str],
        save_name: str = "attention_comparison.png",
        top_k: int = 20
    ) -> str:
        """
        Compare attention from GAT and GAM models.
        
        Args:
            gat_attention: GAT attention scores
            gam_attention: GAM attention scores
            node_names: Node names
            save_name: Filename to save
            top_k: Number of top nodes to show
            
        Returns:
            Path to saved figure
        """
        # Normalize scores
        gat_norm = gat_attention / (np.max(gat_attention) + 1e-8)
        gam_norm = gam_attention / (np.max(gam_attention) + 1e-8)
        
        # Get top-k from GAT
        top_k_indices = np.argsort(gat_norm)[-top_k:][::-1]
        
        # Prepare data
        indices = range(top_k)
        gat_values = [gat_norm[i] for i in top_k_indices]
        gam_values = [gam_norm[i] for i in top_k_indices]
        labels = [node_names[i].split("->")[-1][:30] if i < len(node_names) else f"Node_{i}" 
                 for i in top_k_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gat_values, width, label='GAT', 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, gam_values, width, label='GAM', 
                      color='coral', alpha=0.8)
        
        ax.set_xlabel('Methods', fontsize=12)
        ax.set_ylabel('Normalized Attention Score', fontsize=12)
        ax.set_title('GAT vs GAM Attention Comparison (Top-20)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention comparison to {save_path}")
        return str(save_path)
    
    def plot_method_class_attention(
        self,
        method_scores: Dict[str, float],
        class_scores: Dict[str, float],
        save_name: str = "method_class_attention.png"
    ) -> str:
        """
        Plot method-level and class-level attention hierarchically.
        
        Args:
            method_scores: Method -> score mapping
            class_scores: Class -> score mapping
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Class-level attention
        sorted_classes = sorted(class_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:15]
        class_names = [c.split("/")[-1].replace(";", "") for c, _ in sorted_classes]
        class_values = [s for _, s in sorted_classes]
        
        axes[0].barh(class_names, class_values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Class Attention Score', fontsize=12)
        axes[0].set_title('Top-15 Classes by Attention', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Method-level attention
        sorted_methods = sorted(method_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:20]
        method_names = [m.split("->")[-1][:40] for m, _ in sorted_methods]
        method_values = [s for _, s in sorted_methods]
        
        axes[1].barh(method_names, method_values, color='coral', alpha=0.7)
        axes[1].set_xlabel('Method Attention Score', fontsize=12)
        axes[1].set_title('Top-20 Methods by Attention', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved method-class attention to {save_path}")
        return str(save_path)


class GraphVisualizer:
    """
    Visualizes API call graphs with attention overlays.
    """
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """
        Initialize graph visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_graph_with_attention(
        self,
        graph: nx.DiGraph,
        node_attention: np.ndarray,
        save_name: str = "graph_attention.png",
        max_nodes: int = 50,
        layout: str = "spring"
    ) -> str:
        """
        Visualize graph with attention-based node coloring.
        
        Args:
            graph: NetworkX graph
            node_attention: Node attention scores
            save_name: Filename to save
            max_nodes: Maximum nodes to display
            layout: Graph layout algorithm
            
        Returns:
            Path to saved figure
        """
        # Sample graph if too large
        if graph.number_of_nodes() > max_nodes:
            # Get top-k nodes by attention
            top_k_indices = np.argsort(node_attention)[-max_nodes:]
            nodes = list(graph.nodes())
            top_nodes = [nodes[i] for i in top_k_indices if i < len(nodes)]
            graph = graph.subgraph(top_nodes).copy()
            node_attention = node_attention[top_k_indices]
        
        # Create layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=2, iterations=50)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.circular_layout(graph)
        
        # Normalize attention for coloring
        norm_attention = node_attention / (np.max(node_attention) + 1e-8)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw nodes with attention-based coloring
        nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_color=norm_attention,
            node_size=300,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            width=0.5,
            alpha=0.3,
            ax=ax
        )
        
        # Add colorbar
        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label('Attention Score', fontsize=12)
        
        ax.set_title('API Call Graph with Attention Overlay',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved graph visualization to {save_path}")
        return str(save_path)
    
    def plot_subgraph_around_node(
        self,
        graph: nx.DiGraph,
        target_node: str,
        node_attention: np.ndarray,
        save_name: str = "subgraph.png",
        depth: int = 2
    ) -> str:
        """
        Visualize k-hop neighborhood around a target node.
        
        Args:
            graph: NetworkX graph
            target_node: Node to center on
            node_attention: Node attention scores
            save_name: Filename to save
            depth: Number of hops
            
        Returns:
            Path to saved figure
        """
        # Get k-hop subgraph
        subgraph_nodes = set([target_node])
        current_nodes = set([target_node])
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                if node in graph:
                    # Add successors and predecessors
                    next_nodes.update(graph.successors(node))
                    next_nodes.update(graph.predecessors(node))
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        # Get attention for subgraph nodes
        node_list = list(graph.nodes())
        subgraph_attention = []
        for node in subgraph.nodes():
            if node in node_list:
                idx = node_list.index(node)
                subgraph_attention.append(node_attention[idx])
            else:
                subgraph_attention.append(0)
        
        subgraph_attention = np.array(subgraph_attention)
        
        # Plot
        return self.plot_graph_with_attention(
            subgraph, subgraph_attention, save_name, 
            max_nodes=100, layout="kamada_kawai"
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test attention visualizer
    visualizer = AttentionVisualizer(output_dir="test_viz")
    
    # Generate dummy data
    num_nodes = 100
    attention_scores = np.random.exponential(0.001, num_nodes)
    node_names = [f"Lcom/example/Class{i//10};->method{i}()V" 
                  for i in range(num_nodes)]
    
    # Create visualizations
    visualizer.plot_attention_heatmap(
        attention_scores, node_names, top_k=20
    )
    
    visualizer.plot_attention_distribution(attention_scores)
    
    # GAT vs GAM comparison
    gam_attention = np.random.exponential(0.0012, num_nodes)
    visualizer.plot_attention_comparison(
        attention_scores, gam_attention, node_names
    )
    
    print("✅ Test visualizations created in test_viz/")