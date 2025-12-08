"""
Upgraded APICallGraphBuilder (compatible with upgraded SensitiveAPIFilter)

Features:
- Uses filtered API data when available (sensitive-only)
- Uses api_filter.create_api_features(api) to populate API node metadata
- Reflection node detection + edges (connects reflection -> api nodes where possible)
- Uses decompiler-provided method_calls (preferred) for accurate method->method edges
- Safe heuristics fallback if method_calls absent
- Uses `class_name` attribute to avoid Python keyword conflict
- Defensive logging and per-apk stats support
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


class APICallGraphBuilder:
    def __init__(self, api_filter, reports_dir: str = "data/reports"):
        """
        Args:
            api_filter: instance of upgraded SensitiveAPIFilter (expects create_api_features())
            reports_dir: directory to write per-apk graph reports for debugging
        """
        self.api_filter = api_filter
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def build_graph(self, decompiled_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a directed API-call graph from decompiled APK data.

        decompiled_data expected keys:
          - methods: list of method dicts (name, class, access_flags, code_size, dex)
          - classes: mapping class -> list(method_full_names)
          - api_calls: method -> [raw_api_signatures]
          - reflection_evidence: method -> [notes...]
          - method_calls: optional: method -> [callee_method_full_names]
          - filtered_apis: optional: result of SensitiveAPIFilter.filter_api_calls(decompiled_result)
        """

        apk_name = decompiled_data.get("apk_name", "unknown")
        graph = nx.DiGraph()

        if not decompiled_data.get("success", False):
            logger.error(f"[{apk_name}] decompilation unsuccessful - skipping graph build")
            return graph

        methods = decompiled_data.get("methods", []) or []
        api_calls = decompiled_data.get("api_calls", {}) or {}
        classes = decompiled_data.get("classes", {}) or {}
        reflection = decompiled_data.get("reflection_evidence", {}) or {}
        method_calls = decompiled_data.get("method_calls", {}) or {}
        filtered = decompiled_data.get("filtered_apis") or decompiled_data.get("filtered_data") or {}

        logger.info(f"[{apk_name}] Building graph: methods={len(methods)}, classes={len(classes)}")

        # 1) Add method nodes
        self._add_method_nodes(graph, methods, api_calls)

        # 2) Add API nodes (prefer filtered data)
        if filtered and filtered.get("api_stats"):
            self._add_api_nodes_from_filtered(graph, filtered)
        else:
            self._add_api_nodes(graph, api_calls)

        # 3) Add method -> api edges
        if filtered and filtered.get("methods"):
            self._add_call_edges_from_filtered(graph, filtered)
        else:
            self._add_call_edges(graph, api_calls)

        # 4) Add method -> method edges (prefer method_calls from decompiler)
        if method_calls:
            self._add_method_to_method_edges_from_decompiler(graph, method_calls)
        else:
            self._add_method_to_method_edges_heuristic(graph, api_calls)

        # 5) Add reflection nodes & edges
        self._add_reflection_nodes_and_edges(graph, reflection)

        # 6) graph metadata
        self._add_graph_metadata(graph, decompiled_data)

        # 7) prune to sensitive neighborhood
        pruned = self._prune_graph(graph)
        # 8) write small debug report
        try:
            rpt = self.get_graph_statistics(pruned)
            rpt.update({
                "apk_name": apk_name,
                "num_methods_total": len(methods),
                "num_raw_api_methods": len(api_calls),
                "num_filtered_api_methods": len(filtered.get("methods", {})) if filtered else 0,
            })
            with open(self.reports_dir / f"{apk_name}.graph_report.json", "w") as rf:
                json.dump(rpt, rf, indent=2)
        except Exception:
            logger.debug(f"[{apk_name}] failed to write report", exc_info=True)

        logger.info(f"[{apk_name}] Graph built: nodes={pruned.number_of_nodes()} edges={pruned.number_of_edges()}")
        return pruned

    # -------------------------------------------------------------------------
    # Node helpers
    # -------------------------------------------------------------------------
    def _add_method_nodes(self, graph: nx.DiGraph, methods: List[Dict[str, Any]], api_calls: Dict[str, List[str]]) -> None:
        for m in methods:
            name = m.get("name")
            if not name:
                continue
            apis = api_calls.get(name, []) or []
            sens_count = sum(1 for a in apis if self.api_filter.is_sensitive(a))
            graph.add_node(
                name,
                type="method",
                class_name=m.get("class", ""),
                access_flags=m.get("access_flags"),
                code_size=m.get("code_size", 0),
                dex=m.get("dex"),
                api_count=len(apis),
                sensitive_api_count=sens_count,
                has_sensitive_api=(sens_count > 0)
            )

    def _add_api_nodes(self, graph: nx.DiGraph, api_calls: Dict[str, List[str]]) -> None:
        all_apis = set()
        for lst in api_calls.values():
            if lst:
                all_apis.update(lst)
        for api in all_apis:
            if not self.api_filter.is_sensitive(api):
                continue
            # prefer using create_api_features if extractor provides it
            try:
                features = self.api_filter.create_api_features(api)
                category = features.get("category", "UNKNOWN")
                category_id = features.get("category_id", -1)
            except AttributeError:
                # fallback: try a mapping or use stable hash
                if hasattr(self.api_filter, "api_to_category"):
                    category = self.api_filter.api_to_category.get(api, "UNKNOWN")
                else:
                    category = "UNKNOWN"
                category_id = abs(hash(category)) % 10000

            graph.add_node(
                api,
                type="api",
                is_sensitive=True,
                category=category,
                category_id=category_id,
                class_name=api.split("->")[0] if "->" in api else "",
                api_count=0,
                sensitive_api_count=0
            )

    def _add_api_nodes_from_filtered(self, graph: nx.DiGraph, filtered: Dict[str, Any]) -> None:
        api_stats = filtered.get("api_stats", {}) or {}
        for api, info in api_stats.items():
            # info expected to contain 'category'
            category = info.get("category", "UNKNOWN")
            category_id = self.api_filter.get_category_id(category) if hasattr(self.api_filter, "get_category_id") else abs(hash(category)) % 10000
            graph.add_node(
                api,
                type="api",
                is_sensitive=True,
                category=category,
                category_id=category_id,
                class_name=api.split("->")[0] if "->" in api else "",
                api_count=0,
                sensitive_api_count=0
            )

    # -------------------------------------------------------------------------
    # Edges: method -> api
    # -------------------------------------------------------------------------
    def _add_call_edges(self, graph: nx.DiGraph, api_calls: Dict[str, List[str]]) -> None:
        for caller, lst in (api_calls or {}).items():
            if caller not in graph:
                continue
            for api in (lst or []):
                if api in graph:
                    graph.add_edge(caller, api, call_type="direct", is_sensitive_call=self.api_filter.is_sensitive(api))
                    # update method meta
                    graph.nodes[caller]["api_count"] = graph.nodes[caller].get("api_count", 0) + 1
                    if self.api_filter.is_sensitive(api):
                        graph.nodes[caller]["sensitive_api_count"] = graph.nodes[caller].get("sensitive_api_count", 0) + 1
                        graph.nodes[caller]["has_sensitive_api"] = True

    def _add_call_edges_from_filtered(self, graph: nx.DiGraph, filtered: Dict[str, Any]) -> None:
        methods_map = filtered.get("methods", {}) or {}
        for method, apis in methods_map.items():
            if method not in graph:
                continue
            for api in (apis or []):
                if api in graph:
                    graph.add_edge(method, api, call_type="direct", is_sensitive_call=True)
                    graph.nodes[method]["api_count"] = graph.nodes[method].get("api_count", 0) + 1
                    graph.nodes[method]["sensitive_api_count"] = graph.nodes[method].get("sensitive_api_count", 0) + 1
                    graph.nodes[method]["has_sensitive_api"] = True

    # -------------------------------------------------------------------------
    # Method->method edges: prefer real method_calls from decompiler
    # -------------------------------------------------------------------------
    def _add_method_to_method_edges_from_decompiler(self, graph: nx.DiGraph, method_calls: Dict[str, List[str]]) -> None:
        for caller, callees in (method_calls or {}).items():
            if caller not in graph:
                continue
            for callee in (callees or []):
                if callee in graph:
                    graph.add_edge(caller, callee, call_type="method_call", is_sensitive_call=False)

    def _add_method_to_method_edges_heuristic(self, graph: nx.DiGraph, api_calls: Dict[str, List[str]]) -> None:
        # build class -> methods
        class_map = defaultdict(list)
        for node, data in graph.nodes(data=True):
            if data.get("type") == "method":
                class_map[data.get("class_name", "")].append(node)

        # connect methods in small classes conservatively
        for cls, methods in class_map.items():
            if not methods or len(methods) > 40:
                continue
            for i in range(len(methods) - 1):
                m1, m2 = methods[i], methods[i+1]
                # connect only if they share at least one sensitive API successor
                succ1 = {n for n in graph.successors(m1) if graph.nodes[n].get("type") == "api"}
                succ2 = {n for n in graph.successors(m2) if graph.nodes[n].get("type") == "api"}
                if succ1 and succ2 and (succ1 & succ2):
                    graph.add_edge(m1, m2, call_type="inferred", is_sensitive_call=False)

    # -------------------------------------------------------------------------
    # Reflection handling
    # -------------------------------------------------------------------------
    def _add_reflection_nodes_and_edges(self, graph: nx.DiGraph, reflection: Dict[str, List[str]]) -> None:
        for method, notes in (reflection or {}).items():
            if method not in graph:
                # defensive: add missing method node
                graph.add_node(method, type="method", class_name="", access_flags=None, code_size=0, api_count=0, sensitive_api_count=0, has_sensitive_api=False)
            for i, note in enumerate(notes):
                safe_note = note.replace("/", "_").replace(" ", "_").replace("'", "").replace('"', '')
                node_name = f"REFLECTION::{method}::{i}::{safe_note}"
                graph.add_node(node_name, type="reflection", is_sensitive=True, category="reflection", category_id=-1, note=note)
                graph.add_edge(method, node_name, call_type="reflection", is_sensitive_call=True)
                # attempt to connect reflection node to API nodes if note includes class info
                try:
                    if "class='" in note:
                        cls = note.split("class='", 1)[1].split("'", 1)[0]
                        dalvik = "L" + cls.replace(".", "/") + ";"
                        # connect to API nodes with matching class_name suffix
                        for n, d in graph.nodes(data=True):
                            if d.get("type") == "api":
                                api_cls = d.get("class_name", "")
                                if api_cls and (api_cls == dalvik or api_cls.endswith(dalvik.split("/")[-1] + ";")):
                                    graph.add_edge(node_name, n, call_type="reflects_to_api", is_sensitive_call=True)
                except Exception:
                    continue

    # -------------------------------------------------------------------------
    # Metadata, pruning, stats
    # -------------------------------------------------------------------------
    def _add_graph_metadata(self, graph: nx.DiGraph, data: Dict[str, Any]) -> None:
        graph.graph["apk_name"] = data.get("apk_name", "unknown")
        graph.graph["package_name"] = data.get("package_name")
        graph.graph["label"] = data.get("label", "unknown")
        graph.graph["num_methods"] = len(data.get("methods", []))
        graph.graph["num_classes"] = len(data.get("classes", {}))

    def _prune_graph(self, graph: nx.DiGraph, keep_unconnected: bool = False) -> nx.DiGraph:
        if graph.number_of_nodes() == 0:
            return graph

        sensitive_nodes = {
            n for n, d in graph.nodes(data=True)
            if (d.get("type") == "api" and d.get("is_sensitive", False))
               or d.get("has_sensitive_api", False)
               or d.get("type") == "reflection"
        }

        if not sensitive_nodes:
            logger.warning("No sensitive nodes found in graph; returning original graph")
            return graph

        keep = set(sensitive_nodes)
        for n in list(sensitive_nodes):
            keep.update(graph.predecessors(n))
            keep.update(graph.successors(n))

        pruned = graph.subgraph(keep).copy()
        if not keep_unconnected:
            isolated = list(nx.isolates(pruned))
            if isolated:
                pruned.remove_nodes_from(isolated)
        logger.debug(f"Pruned graph: {graph.number_of_nodes()} -> {pruned.number_of_nodes()}")
        return pruned

    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_method_nodes": sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "method"),
            "num_api_nodes": sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "api"),
            "num_reflection_nodes": sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "reflection"),
            "num_sensitive_apis": sum(1 for _, d in graph.nodes(data=True) if d.get("is_sensitive", False)),
            "is_weakly_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
            "num_components": nx.number_weakly_connected_components(graph) if graph.number_of_nodes() > 0 else 0
        }
        return stats
