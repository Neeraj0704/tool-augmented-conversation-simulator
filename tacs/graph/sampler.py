from __future__ import annotations

import logging
import random

import networkx as nx
from pydantic import BaseModel

from tacs.config import config
from tacs.graph.models import EdgeType, NodeType, node_id

logger = logging.getLogger(__name__)


class ToolChain(BaseModel):
    """A sampled chain of tool calls to be executed in a conversation."""

    steps: list[list[str]]  # each step is a list of endpoint node IDs
    pattern: str            # "multi_step", "parallel", "hybrid"
    tool_ids: list[str]     # distinct tool IDs in traversal order


class ToolChainSampler:
    """Samples realistic tool chains from a Tool Graph."""

    def __init__(self, graph: nx.DiGraph, seed: int) -> None:
        self._graph = graph
        self._rng = random.Random(seed)

    def sample(self, pattern: str = "multi_step", min_steps: int = 3) -> ToolChain:
        """Sample a tool chain with the given pattern."""
        if min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if pattern == "multi_step":
            return self._sample_multi_step(min_steps)
        elif pattern == "parallel":
            return self._sample_parallel(min_steps)
        elif pattern == "hybrid":
            return self._sample_hybrid()
        else:
            raise ValueError(f"Unknown pattern: {pattern!r}")

    # --- Pattern implementations ---

    def _sample_multi_step(self, min_steps: int) -> ToolChain:
        """Sample a sequential chain — strict COMPATIBLE_WITH first, loose fallback."""
        for _ in range(config.max_retries):
            chain = self._strict_multi_step(min_steps)
            if chain:
                return chain
            chain = self._loose_multi_step(min_steps)
            if chain:
                return chain

        raise ValueError(
            f"Failed to sample multi_step chain of length >= {min_steps} "
            f"after {config.max_retries} retries."
        )

    def _strict_multi_step(self, min_steps: int) -> ToolChain | None:
        """Follow COMPATIBLE_WITH edges greedily from a random start endpoint."""
        candidates = [
            nid for nid in self._all_endpoints()
            if self._compatible_neighbors(nid, set())
        ]
        if not candidates:
            return None

        start = self._rng.choice(candidates)
        steps: list[list[str]] = [[start]]
        used: set[str] = {start}

        while True:
            neighbors = self._compatible_neighbors(steps[-1][0], used)
            if not neighbors:
                break
            nxt = self._rng.choice(neighbors)
            steps.append([nxt])
            used.add(nxt)

        if len(steps) < min_steps:
            return None

        tool_ids = self._extract_tool_ids(steps)
        if len(set(tool_ids)) < 2:
            return None

        logger.debug("Strict multi_step chain built: %d steps", len(steps))
        return ToolChain(steps=steps, pattern="multi_step", tool_ids=tool_ids)

    def _loose_multi_step(self, min_steps: int) -> ToolChain | None:
        """Build a chain from a random concept with endpoints from >= 2 tools."""
        concepts = self._concept_nodes()
        if not concepts:
            return None

        self._rng.shuffle(concepts)
        for concept_nid in concepts:
            endpoints = self._endpoints_for_concept(concept_nid)
            by_tool: dict[str, list[str]] = {}
            for ep in endpoints:
                by_tool.setdefault(self._tool_id(ep), []).append(ep)

            if len(by_tool) < 2:
                continue

            selected: list[str] = []
            used: set[str] = set()

            # First pick one from each tool to guarantee diversity
            tools = list(by_tool.keys())
            self._rng.shuffle(tools)
            for tid in tools:
                if len(selected) >= min_steps:
                    break
                ep = self._rng.choice(by_tool[tid])
                if ep not in used:
                    selected.append(ep)
                    used.add(ep)

            # Fill remaining slots if needed
            all_eps = [ep for eps in by_tool.values() for ep in eps]
            self._rng.shuffle(all_eps)
            for ep in all_eps:
                if len(selected) >= min_steps:
                    break
                if ep not in used:
                    selected.append(ep)
                    used.add(ep)

            if len(selected) < min_steps:
                continue

            steps = [[ep] for ep in selected]
            tool_ids = self._extract_tool_ids(steps)
            if len(set(tool_ids)) < 2:
                continue

            logger.debug("Loose multi_step chain built: %d steps", len(steps))
            return ToolChain(steps=steps, pattern="multi_step", tool_ids=tool_ids)

        return None

    def _sample_parallel(self, min_steps: int) -> ToolChain:
        """Sample a single parallel step with min_steps endpoints from >= 2 tools."""
        for _ in range(config.max_retries):
            concepts = self._concept_nodes()
            if not concepts:
                break

            concept_nid = self._rng.choice(concepts)
            endpoints = self._endpoints_for_concept(concept_nid)
            by_tool: dict[str, list[str]] = {}
            for ep in endpoints:
                by_tool.setdefault(self._tool_id(ep), []).append(ep)

            if len(by_tool) < 2:
                continue

            selected: list[str] = []
            used: set[str] = set()
            tools = list(by_tool.keys())
            self._rng.shuffle(tools)

            # Guarantee at least one from each of two tools
            for tid in tools[:2]:
                ep = self._rng.choice(by_tool[tid])
                selected.append(ep)
                used.add(ep)

            # Fill remaining slots
            all_eps = [ep for eps in by_tool.values() for ep in eps]
            self._rng.shuffle(all_eps)
            for ep in all_eps:
                if len(selected) >= min_steps:
                    break
                if ep not in used:
                    selected.append(ep)
                    used.add(ep)

            if len(selected) < min_steps:
                continue

            steps = [selected[:min_steps]]
            tool_ids = self._extract_tool_ids(steps)
            if len(tool_ids) < 2:
                continue

            logger.debug("Parallel chain built: %d endpoints", len(steps[0]))
            return ToolChain(steps=steps, pattern="parallel", tool_ids=tool_ids)

        raise ValueError(
            f"Failed to sample parallel chain of width >= {min_steps} "
            f"after {config.max_retries} retries."
        )

    def _sample_hybrid(self) -> ToolChain:
        """Sample a hybrid chain: [[ep1], [ep2, ep3_parallel]]."""
        for _ in range(config.max_retries):
            all_eps = self._all_endpoints()
            if not all_eps:
                break

            # Step 1: prefer endpoints with outgoing COMPATIBLE_WITH edges
            candidates = [
                nid for nid in all_eps
                if self._compatible_neighbors(nid, set())
            ] or all_eps

            ep1 = self._rng.choice(candidates)
            used: set[str] = {ep1}

            # Step 2: sequential neighbor
            seq_neighbors = self._compatible_neighbors(ep1, used)
            if not seq_neighbors:
                continue

            ep2 = self._rng.choice(seq_neighbors)
            used.add(ep2)

            # Step 2 parallel: find a same-concept endpoint
            ep3 = self._find_parallel_endpoint(ep1, used)
            if ep3 is None:
                continue

            steps = [[ep1], [ep2, ep3]]
            tool_ids = self._extract_tool_ids(steps)
            if len(set(tool_ids)) < 2:
                continue

            logger.debug("Hybrid chain built: steps=%s", steps)
            return ToolChain(steps=steps, pattern="hybrid", tool_ids=tool_ids)

        raise ValueError(
            f"Failed to sample hybrid chain after {config.max_retries} retries."
        )

    def _find_parallel_endpoint(self, ep1: str, used: set[str]) -> str | None:
        """Find a concept-compatible endpoint to run in parallel at step 2."""
        tool_nid = node_id(NodeType.TOOL, self._tool_id(ep1))
        for nbr in self._graph.successors(tool_nid):
            if self._graph.nodes[nbr].get("type") != NodeType.CONCEPT:
                continue
            candidates = [
                ep for ep in self._endpoints_for_concept(nbr)
                if ep not in used
            ]
            if candidates:
                return self._rng.choice(candidates)
        return None

    # --- Private helpers ---

    def _all_endpoints(self) -> list[str]:
        """Return all endpoint node IDs in the graph."""
        return [
            nid for nid, data in self._graph.nodes(data=True)
            if data.get("type") == NodeType.ENDPOINT
        ]

    def _tool_id(self, ep_nid: str) -> str:
        """Return the tool_id attribute of an endpoint node."""
        return self._graph.nodes[ep_nid]["tool_id"]

    def _compatible_neighbors(self, ep_nid: str, exclude: set[str]) -> list[str]:
        """Return endpoint node IDs reachable via COMPATIBLE_WITH edges."""
        return [
            nbr for nbr in self._graph.successors(ep_nid)
            if (
                self._graph.edges[ep_nid, nbr].get("type") == EdgeType.COMPATIBLE_WITH
                and self._graph.nodes[nbr].get("type") == NodeType.ENDPOINT
                and nbr not in exclude
            )
        ]

    def _concept_nodes(self) -> list[str]:
        """Return all concept node IDs in the graph."""
        return [
            nid for nid, data in self._graph.nodes(data=True)
            if data.get("type") == NodeType.CONCEPT
        ]

    def _endpoints_for_concept(self, concept_nid: str) -> list[str]:
        """Return all endpoint node IDs reachable through a concept node."""
        endpoints: list[str] = []
        for tool_nid in self._graph.predecessors(concept_nid):
            if self._graph.nodes[tool_nid].get("type") != NodeType.TOOL:
                continue
            for ep_nid in self._graph.successors(tool_nid):
                if self._graph.nodes[ep_nid].get("type") == NodeType.ENDPOINT:
                    endpoints.append(ep_nid)
        return endpoints

    def _extract_tool_ids(self, steps: list[list[str]]) -> list[str]:
        """Extract ordered distinct tool IDs by traversing steps left to right."""
        seen: set[str] = set()
        tool_ids: list[str] = []
        for step in steps:
            for ep_nid in step:
                tid = self._tool_id(ep_nid)
                if tid not in seen:
                    tool_ids.append(tid)
                    seen.add(tid)
        return tool_ids
