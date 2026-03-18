from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx

from tacs.graph.models import EdgeType, NodeType, node_id
from tacs.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

_GRAPH_FILE = "graph.pkl"

_ALIASES: dict[str, list[str]] = {
    "city": ["cityname", "location", "place"],
    "temperature": ["temp", "degrees"],
    "id": ["identifier", "key"],
    "query": ["search", "keyword", "q"],
    "date": ["day", "datetime", "time"],
}

_ALIAS_LOOKUP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _ALIASES.items()
    for alias in aliases
}


def _normalize(name: str) -> str:
    """Lowercase and strip spaces, underscores, and hyphens for fuzzy matching."""
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _candidate_keys(name: str) -> set[str]:
    """Return all normalized name variants for symmetric alias matching."""
    norm = _normalize(name)
    canonical = _ALIAS_LOOKUP.get(norm, norm)
    keys = {norm, canonical}
    keys.update(_ALIASES.get(canonical, []))
    return {_normalize(k) for k in keys}


class ToolGraphBuilder:
    """Builds a NetworkX DiGraph from a ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def build(self) -> nx.DiGraph:
        """Build and return the Tool Graph."""
        graph = nx.DiGraph()

        ep_response_fields: dict[str, list[str]] = {}
        ep_required_params: dict[str, list[str]] = {}

        for tool in self._registry.list_tools():
            tool_nid = node_id(NodeType.TOOL, tool.tool_id)

            # a) Tool node
            graph.add_node(
                tool_nid,
                type=NodeType.TOOL,
                name=tool.name,
                category=tool.category,
            )

            # b) Concept node + TAGGED_WITH edge
            concept_nid = node_id(NodeType.CONCEPT, tool.category)
            if not graph.has_node(concept_nid):
                graph.add_node(concept_nid, type=NodeType.CONCEPT, name=tool.category)
            graph.add_edge(tool_nid, concept_nid, type=EdgeType.TAGGED_WITH)

            for ep in tool.endpoints:
                ep_nid = node_id(NodeType.ENDPOINT, ep.name, parent=tool.tool_id)
                ep_parent = f"{tool.tool_id}.{ep.name}"

                # c) Endpoint node + HAS_ENDPOINT edge
                graph.add_node(
                    ep_nid,
                    type=NodeType.ENDPOINT,
                    name=ep.name,
                    tool_id=tool.tool_id,
                )
                graph.add_edge(tool_nid, ep_nid, type=EdgeType.HAS_ENDPOINT)

                # d) Parameter nodes + HAS_PARAMETER edges
                for p in ep.parameters:
                    p_nid = node_id(NodeType.PARAMETER, p.name, parent=ep_parent)
                    graph.add_node(
                        p_nid,
                        type=NodeType.PARAMETER,
                        name=p.name,
                        required=p.required,
                    )
                    graph.add_edge(ep_nid, p_nid, type=EdgeType.HAS_PARAMETER)

                # e) ResponseField nodes + RETURNS edges
                for field in ep.response_fields:
                    rf_nid = node_id(NodeType.RESPONSE_FIELD, field, parent=ep_parent)
                    graph.add_node(
                        rf_nid,
                        type=NodeType.RESPONSE_FIELD,
                        name=field,
                    )
                    graph.add_edge(ep_nid, rf_nid, type=EdgeType.RETURNS)

                ep_response_fields[ep_nid] = [_normalize(f) for f in ep.response_fields]
                ep_required_params[ep_nid] = [
                    _normalize(p.name) for p in ep.parameters if p.required
                ]

        # f) COMPATIBLE_WITH edges
        compatible_count = self._add_compatibility_edges(
            graph, ep_response_fields, ep_required_params
        )

        self._log_stats(graph, compatible_count)
        return graph

    def _add_compatibility_edges(
        self,
        graph: nx.DiGraph,
        ep_response_fields: dict[str, list[str]],
        ep_params: dict[str, list[str]],
    ) -> int:
        """Add COMPATIBLE_WITH edges using inverted index + alias matching."""
        param_index: dict[str, list[str]] = {}
        for ep_nid, params in ep_params.items():
            for param in params:
                for key in _candidate_keys(param):
                    param_index.setdefault(key, []).append(ep_nid)

        compatible_count = 0
        for src_nid, fields in ep_response_fields.items():
            added: set[str] = set()
            for field in fields:
                candidates: set[str] = set()
                for key in _candidate_keys(field):
                    candidates.update(param_index.get(key, []))
                for dst_nid in candidates:
                    if dst_nid == src_nid:
                        continue
                    if dst_nid in added:
                        continue
                    graph.add_edge(src_nid, dst_nid, type=EdgeType.COMPATIBLE_WITH)
                    added.add(dst_nid)
                    compatible_count += 1

        return compatible_count

    def _log_stats(self, graph: nx.DiGraph, compatible_count: int) -> None:
        """Log graph build statistics."""
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()

        counts_by_type: dict[NodeType, int] = {}
        for _, data in graph.nodes(data=True):
            nt = data.get("type")
            if nt:
                counts_by_type[nt] = counts_by_type.get(nt, 0) + 1

        logger.info(
            "Graph built: %d nodes, %d edges, %d compatible_with edges",
            total_nodes,
            total_edges,
            compatible_count,
        )
        for nt, count in sorted(counts_by_type.items(), key=lambda x: x[0].value):
            logger.info("  %s: %d", nt.value, count)

    @staticmethod
    def save(graph: nx.DiGraph, artifacts_dir: Path) -> None:
        """Pickle the graph to artifacts_dir/graph.pkl."""
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = artifacts_dir / _GRAPH_FILE
            with open(out_path, "wb") as f:
                pickle.dump(graph, f)
            logger.info("Saved graph to %s", out_path)
        except OSError:
            logger.exception("Failed to save graph to %s", artifacts_dir)
            raise

    @staticmethod
    def load(artifacts_dir: Path) -> nx.DiGraph:
        """Load a pickled graph from artifacts_dir/graph.pkl."""
        pkl_path = artifacts_dir / _GRAPH_FILE
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Graph artifact not found at {pkl_path}. Run `tacs build` first."
            )
        try:
            with open(pkl_path, "rb") as f:
                graph = pickle.load(f)
            logger.info(
                "Loaded graph (%d nodes, %d edges) from %s",
                graph.number_of_nodes(),
                graph.number_of_edges(),
                pkl_path,
            )
            return graph
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.exception("Failed to load graph from %s", pkl_path)
            raise
