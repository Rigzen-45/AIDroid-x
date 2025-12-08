"""
Malicious Code Localization Module
Aggregates attention scores: Node → Method → Class
Thresholds: τ_method = 0.0001, τ_class = 0.001
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CodeLocalizer:
    """
    Localizes malicious code at method and class levels using attention scores.
    
    Process:
    1. Node-level attention scores (from GAT/GAM)
    2. Aggregate to method-level scores
    3. Aggregate to class-level scores
    4. Apply thresholds to identify malicious code
    """
    
    def __init__(
        self,
        method_threshold: float = 0.0001,
        class_threshold: float = 0.001,
        min_malicious_methods: int = 1
    ):
        """
        Initialize code localizer.
        
        Args:
            method_threshold: Threshold for flagging malicious methods (τ_method)
            class_threshold: Threshold for flagging malicious classes (τ_class)
            min_malicious_methods: Minimum methods to flag class as malicious
        """
        self.method_threshold = method_threshold
        self.class_threshold = class_threshold
        self.min_malicious_methods = min_malicious_methods
        
        logger.info(f"Initialized CodeLocalizer: τ_method={method_threshold}, "
                   f"τ_class={class_threshold}")
    
    def localize(
        self,
        node_attention: torch.Tensor,
        node_names: List[str],
        graph_data
    ) -> Dict:
        """
        Perform complete code localization from node attention to classes.
        
        Args:
            node_attention: Attention scores for each node [num_nodes]
            node_names: List of node identifiers (method signatures)
            graph_data: PyG Data object with metadata
            
        Returns:
            Dictionary containing:
                - method_scores: Dict[method_name, score]
                - malicious_methods: List[method_name]
                - class_scores: Dict[class_name, score]
                - malicious_classes: List[class_name]
                - localization_report: Detailed report
        """
        # Step 1: Node attention to method attention
        method_scores = self._aggregate_node_to_method(
            node_attention, node_names
        )
        
        # Step 2: Identify malicious methods
        malicious_methods = self._identify_malicious_methods(method_scores)
        
        # Step 3: Method attention to class attention
        class_scores = self._aggregate_method_to_class(
            method_scores, node_names
        )
        
        # Step 4: Identify malicious classes
        malicious_classes = self._identify_malicious_classes(
            class_scores, method_scores
        )
        
        # Step 5: Generate detailed report
        report = self._generate_report(
            method_scores, malicious_methods,
            class_scores, malicious_classes,
            node_names
        )
        
        result = {
            "method_scores": method_scores,
            "malicious_methods": malicious_methods,
            "class_scores": class_scores,
            "malicious_classes": malicious_classes,
            "localization_report": report
        }
        
        logger.info(f"Localization complete: {len(malicious_methods)} malicious methods, "
                   f"{len(malicious_classes)} malicious classes")
        
        return result
    
    def _aggregate_node_to_method(
        self,
        node_attention: torch.Tensor,
        node_names: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate node-level attention to method-level attention.
        
        For each method:
        - If node is a method: use its attention directly
        - If node is an API: aggregate attention to calling methods
        
        Args:
            node_attention: Attention scores [num_nodes]
            node_names: Node identifiers
            
        Returns:
            Dictionary mapping method names to aggregated attention scores
        """
        method_scores = defaultdict(float)
        method_counts = defaultdict(int)
        
        # Convert to numpy for easier manipulation
        if isinstance(node_attention, torch.Tensor):
            attention_array = node_attention.cpu().numpy()
        else:
            attention_array = np.array(node_attention)
        
        for node_idx, node_name in enumerate(node_names):
            attention_score = float(attention_array[node_idx])
            
            # Extract method name (before -> for method nodes)
            if "->" in node_name:
                # This is a method or API call
                # For methods: use the full signature
                # For APIs: attribute to the method that calls it (handled separately)
                method_name = node_name
                method_scores[method_name] += attention_score
                method_counts[method_name] += 1
        
        # Average attention per method
        for method in method_scores:
            if method_counts[method] > 0:
                method_scores[method] /= method_counts[method]
        
        return dict(method_scores)
    
    def _identify_malicious_methods(
        self,
        method_scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Identify methods with attention above threshold.
        
        Args:
            method_scores: Dictionary of method -> score
            
        Returns:
            List of (method_name, score) tuples for malicious methods
        """
        malicious = []
        
        for method, score in method_scores.items():
            if score >= self.method_threshold:
                malicious.append((method, score))
        
        # Sort by score descending
        malicious.sort(key=lambda x: x[1], reverse=True)
        
        return malicious
    
    def _aggregate_method_to_class(
        self,
        method_scores: Dict[str, float],
        node_names: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate method-level attention to class-level attention.
        
        Class score = average of all method scores in that class
        
        Args:
            method_scores: Dictionary of method -> score
            node_names: Node identifiers for extracting class info
            
        Returns:
            Dictionary mapping class names to aggregated scores
        """
        class_method_scores = defaultdict(list)
        
        # Group methods by class
        for method, score in method_scores.items():
            if "->" in method:
                # Extract class name (before ->)
                class_name = method.split("->")[0]
                class_method_scores[class_name].append(score)
        
        # Compute class scores (average of method scores)
        class_scores = {}
        for class_name, scores in class_method_scores.items():
            if scores:
                # Use average or max (paper uses average)
                class_scores[class_name] = np.mean(scores)
        
        return class_scores
    
    def _identify_malicious_classes(
        self,
        class_scores: Dict[str, float],
        method_scores: Dict[str, float]
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Identify classes with:
        1. Aggregated score >= class_threshold
        2. At least min_malicious_methods with score >= method_threshold
        
        Args:
            class_scores: Dictionary of class -> score
            method_scores: Dictionary of method -> score
            
        Returns:
            List of (class_name, score, malicious_methods) tuples
        """
        malicious_classes = []
        
        for class_name, class_score in class_scores.items():
            # Check class threshold
            if class_score < self.class_threshold:
                continue
            
            # Find malicious methods in this class
            class_malicious_methods = []
            for method, method_score in method_scores.items():
                if "->" in method:
                    method_class = method.split("->")[0]
                    if method_class == class_name:
                        if method_score >= self.method_threshold:
                            class_malicious_methods.append(method)
            
            # Check minimum malicious methods requirement
            if len(class_malicious_methods) >= self.min_malicious_methods:
                malicious_classes.append((
                    class_name,
                    class_score,
                    class_malicious_methods
                ))
        
        # Sort by score descending
        malicious_classes.sort(key=lambda x: x[1], reverse=True)
        
        return malicious_classes
    
    def _generate_report(
        self,
        method_scores: Dict[str, float],
        malicious_methods: List[Tuple[str, float]],
        class_scores: Dict[str, float],
        malicious_classes: List[Tuple[str, float, List[str]]],
        node_names: List[str]
    ) -> Dict:
        """
        Generate detailed localization report.
        
        Args:
            method_scores: All method scores
            malicious_methods: Identified malicious methods
            class_scores: All class scores
            malicious_classes: Identified malicious classes
            node_names: Node identifiers
            
        Returns:
            Detailed report dictionary
        """
        report = {
            "summary": {
                "total_methods": len(method_scores),
                "total_classes": len(class_scores),
                "malicious_methods_count": len(malicious_methods),
                "malicious_classes_count": len(malicious_classes)
            },
            "malicious_classes": [],
            "top_methods": malicious_methods[:10],  # Top 10 methods
            "statistics": {
                "method_score_mean": float(np.mean(list(method_scores.values()))) if method_scores else 0,
                "method_score_max": float(np.max(list(method_scores.values()))) if method_scores else 0,
                "class_score_mean": float(np.mean(list(class_scores.values()))) if class_scores else 0,
                "class_score_max": float(np.max(list(class_scores.values()))) if class_scores else 0
            }
        }
        
        # Detailed class information
        for class_name, class_score, methods in malicious_classes:
            class_info = {
                "class_name": class_name,
                "score": float(class_score),
                "malicious_methods": [
                    {
                        "method": method.split("->")[1] if "->" in method else method,
                        "full_signature": method,
                        "score": float(method_scores.get(method, 0))
                    }
                    for method in methods
                ],
                "num_malicious_methods": len(methods)
            }
            report["malicious_classes"].append(class_info)
        
        return report
    
    def format_report(self, localization_result: Dict) -> str:
        """
        Format localization result as human-readable string.
        
        Args:
            localization_result: Output from localize()
            
        Returns:
            Formatted report string
        """
        report = localization_result["localization_report"]
        output = []
        
        output.append("=" * 80)
        output.append("MALICIOUS CODE LOCALIZATION REPORT")
        output.append("=" * 80)
        
        # Summary
        summary = report["summary"]
        output.append(f"\nSummary:")
        output.append(f"  Total Methods: {summary['total_methods']}")
        output.append(f"  Total Classes: {summary['total_classes']}")
        output.append(f"  Malicious Methods: {summary['malicious_methods_count']}")
        output.append(f"  Malicious Classes: {summary['malicious_classes_count']}")
        
        # Statistics
        stats = report["statistics"]
        output.append(f"\nStatistics:")
        output.append(f"  Method Score - Mean: {stats['method_score_mean']:.6f}, "
                     f"Max: {stats['method_score_max']:.6f}")
        output.append(f"  Class Score - Mean: {stats['class_score_mean']:.6f}, "
                     f"Max: {stats['class_score_max']:.6f}")
        
        # Malicious classes
        output.append(f"\n{'=' * 80}")
        output.append("MALICIOUS CLASSES")
        output.append("=" * 80)
        
        if report["malicious_classes"]:
            for i, class_info in enumerate(report["malicious_classes"], 1):
                output.append(f"\n[{i}] {class_info['class_name']}")
                output.append(f"    Class Score: {class_info['score']:.6f}")
                output.append(f"    Malicious Methods: {class_info['num_malicious_methods']}")
                
                output.append(f"\n    Methods:")
                for method_info in class_info["malicious_methods"][:5]:  # Top 5
                    output.append(f"      ✗ {method_info['method']}")
                    output.append(f"        Score: {method_info['score']:.6f}")
                
                if class_info['num_malicious_methods'] > 5:
                    remaining = class_info['num_malicious_methods'] - 5
                    output.append(f"      ... and {remaining} more methods")
        else:
            output.append("\nNo malicious classes identified.")
        
        # Top methods across all classes
        output.append(f"\n{'=' * 80}")
        output.append("TOP MALICIOUS METHODS (All Classes)")
        output.append("=" * 80)
        
        for i, (method, score) in enumerate(report["top_methods"], 1):
            method_name = method.split("->")[1] if "->" in method else method
            class_name = method.split("->")[0] if "->" in method else "Unknown"
            output.append(f"\n[{i}] {method_name}")
            output.append(f"    Class: {class_name}")
            output.append(f"    Score: {score:.6f}")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)


class EnsembleLocalizer:
    """
    Combines localization results from multiple models (GAT + GAM).
    Uses ensemble rule: GAM ∧ GAT (both must agree).
    """
    
    def __init__(
        self,
        gat_localizer: CodeLocalizer,
        gam_localizer: CodeLocalizer,
        ensemble_rule: str = "AND"
    ):
        """
        Initialize ensemble localizer.
        
        Args:
            gat_localizer: CodeLocalizer for GAT model
            gam_localizer: CodeLocalizer for GAM model
            ensemble_rule: "AND" or "OR" for combining predictions
        """
        self.gat_localizer = gat_localizer
        self.gam_localizer = gam_localizer
        self.ensemble_rule = ensemble_rule.upper()
        
        logger.info(f"Initialized EnsembleLocalizer with rule: {ensemble_rule}")
    
    def localize(
        self,
        gat_attention: torch.Tensor,
        gam_attention: torch.Tensor,
        node_names: List[str],
        graph_data
    ) -> Dict:
        """
        Perform ensemble localization using both models.
        
        Args:
            gat_attention: Attention from GAT model
            gam_attention: Attention from GAM model
            node_names: Node identifiers
            graph_data: Graph metadata
            
        Returns:
            Combined localization result
        """
        # Get individual localizations
        gat_result = self.gat_localizer.localize(
            gat_attention, node_names, graph_data
        )
        gam_result = self.gam_localizer.localize(
            gam_attention, node_names, graph_data
        )
        
        # Combine results based on ensemble rule
        if self.ensemble_rule == "AND":
            combined = self._combine_and(gat_result, gam_result)
        elif self.ensemble_rule == "OR":
            combined = self._combine_or(gat_result, gam_result)
        else:
            raise ValueError(f"Unknown ensemble rule: {self.ensemble_rule}")
        
        # Add individual results for comparison
        combined["gat_only"] = gat_result
        combined["gam_only"] = gam_result
        
        return combined
    
    def _combine_and(self, gat_result: Dict, gam_result: Dict) -> Dict:
        """
        Combine using AND rule: both models must agree.
        
        Args:
            gat_result: GAT localization
            gam_result: GAM localization
            
        Returns:
            Combined result (intersection)
        """
        # Find methods flagged by both models
        gat_methods = set(m[0] for m in gat_result["malicious_methods"])
        gam_methods = set(m[0] for m in gam_result["malicious_methods"])
        common_methods = gat_methods & gam_methods
        
        # Find classes flagged by both models
        gat_classes = set(c[0] for c in gat_result["malicious_classes"])
        gam_classes = set(c[0] for c in gam_result["malicious_classes"])
        common_classes = gat_classes & gam_classes
        
        # Build combined result
        combined_malicious_methods = [
            (method, (gat_result["method_scores"][method] + 
                     gam_result["method_scores"][method]) / 2)
            for method in common_methods
        ]
        combined_malicious_methods.sort(key=lambda x: x[1], reverse=True)
        
        combined_malicious_classes = []
        for class_name in common_classes:
            # Get class info from both models
            gat_class = next((c for c in gat_result["malicious_classes"] 
                            if c[0] == class_name), None)
            gam_class = next((c for c in gam_result["malicious_classes"] 
                            if c[0] == class_name), None)
            
            if gat_class and gam_class:
                avg_score = (gat_class[1] + gam_class[1]) / 2
                # Combine method lists
                combined_methods = list(set(gat_class[2] + gam_class[2]))
                combined_malicious_classes.append((
                    class_name, avg_score, combined_methods
                ))
        
        combined_malicious_classes.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "method_scores": gat_result["method_scores"],  # Use GAT scores
            "malicious_methods": combined_malicious_methods,
            "class_scores": gat_result["class_scores"],
            "malicious_classes": combined_malicious_classes,
            "ensemble_agreement": {
                "gat_methods": len(gat_methods),
                "gam_methods": len(gam_methods),
                "common_methods": len(common_methods),
                "gat_classes": len(gat_classes),
                "gam_classes": len(gam_classes),
                "common_classes": len(common_classes)
            }
        }
    
    def _combine_or(self, gat_result: Dict, gam_result: Dict) -> Dict:
        """
        Combine using OR rule: either model can flag.
        
        Args:
            gat_result: GAT localization
            gam_result: GAM localization
            
        Returns:
            Combined result (union)
        """
        # Union of flagged methods
        all_methods = set()
        all_methods.update(m[0] for m in gat_result["malicious_methods"])
        all_methods.update(m[0] for m in gam_result["malicious_methods"])
        
        combined_malicious_methods = []
        for method in all_methods:
            gat_score = gat_result["method_scores"].get(method, 0)
            gam_score = gam_result["method_scores"].get(method, 0)
            avg_score = (gat_score + gam_score) / 2
            combined_malicious_methods.append((method, avg_score))
        
        combined_malicious_methods.sort(key=lambda x: x[1], reverse=True)
        
        # Similar for classes
        all_classes = set()
        all_classes.update(c[0] for c in gat_result["malicious_classes"])
        all_classes.update(c[0] for c in gam_result["malicious_classes"])
        
        # ... (similar aggregation for classes)
        
        return {
            "method_scores": gat_result["method_scores"],
            "malicious_methods": combined_malicious_methods,
            "class_scores": gat_result["class_scores"],
            "malicious_classes": [],  # Simplified for brevity
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test data
    node_attention = torch.tensor([
        0.0005, 0.0002, 0.0001, 0.0003, 0.0004,
        0.00001, 0.00002, 0.000015, 0.00003, 0.00001
    ])
    
    node_names = [
        "Lcom/example/MainActivity;->onCreate()V",
        "Lcom/example/MainActivity;->sendData()V",
        "Lcom/example/NetworkService;->uploadData()V",
        "Lcom/example/NetworkService;->connectToServer()V",
        "Lcom/example/Utils;->encryptData([B)[B",
        "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;",
        "Landroid/location/LocationManager;->getLastKnownLocation()Landroid/location/Location;",
        "Ljava/net/HttpURLConnection;->getInputStream()Ljava/io/InputStream;",
        "Landroid/telephony/SmsManager;->sendTextMessage()V",
        "Ljava/lang/String;->toString()Ljava/lang/String;"
    ]
    
    # Initialize localizer
    localizer = CodeLocalizer(
        method_threshold=0.0001,
        class_threshold=0.001
    )
    
    # Perform localization
    result = localizer.localize(node_attention, node_names, None)
    
    # Print report
    report = localizer.format_report(result)
    print(report)