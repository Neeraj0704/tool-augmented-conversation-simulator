"""Unit tests for Tool Graph construction and ToolChainSampler."""
from __future__ import annotations

import pytest
import networkx as nx

from tacs.graph.builder import ToolGraphBuilder
from tacs.graph.models import EdgeType, NodeType, node_id
from tacs.graph.sampler import ToolChain, ToolChainSampler
from tacs.registry.models import Endpoint, Parameter, Tool
from tacs.registry.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    """Registry with two tools that can form a COMPATIBLE_WITH chain.

    weather_api.get_forecast returns 'city' → maps_api.search_places requires 'city'
    so COMPATIBLE_WITH edge should be created.
    """
    tools = [
        Tool(
            tool_id="weather_api",
            name="Weather API",
            category="travel",
            endpoints=[
                Endpoint(
                    name="get_forecast",
                    parameters=[Parameter(name="city", type="string", required=True)],
                    response_fields=["city", "temperature", "humidity"],
                ),
                Endpoint(
                    name="get_historical",
                    parameters=[Parameter(name="city", type="string", required=True)],
                    response_fields=["city", "date", "avg_temp"],
                ),
            ],
        ),
        Tool(
            tool_id="maps_api",
            name="Maps API",
            category="travel",
            endpoints=[
                Endpoint(
                    name="search_places",
                    parameters=[Parameter(name="city", type="string", required=True)],
                    response_fields=["place_id", "name", "address"],
                ),
                Endpoint(
                    name="get_directions",
                    parameters=[
                        Parameter(name="origin", type="string", required=True),
                        Parameter(name="destination", type="string", required=True),
                    ],
                    response_fields=["distance", "duration", "steps"],
                ),
            ],
        ),
        Tool(
            tool_id="booking_api",
            name="Booking API",
            category="travel",
            endpoints=[
                Endpoint(
                    name="book_hotel",
                    parameters=[Parameter(name="city", type="string", required=True)],
                    response_fields=["booking_id", "hotel_name", "price"],
                ),
            ],
        ),
    ]
    return ToolRegistry(tools)


@pytest.fixture
def registry() -> ToolRegistry:
    return _make_registry()


@pytest.fixture
def graph(registry: ToolRegistry) -> nx.DiGraph:
    return ToolGraphBuilder(registry).build()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_graph_is_digraph(self, graph: nx.DiGraph):
        assert isinstance(graph, nx.DiGraph)

    def test_has_tool_nodes(self, graph: nx.DiGraph):
        tool_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.TOOL
        ]
        assert len(tool_nodes) == 3

    def test_has_endpoint_nodes(self, graph: nx.DiGraph):
        ep_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.ENDPOINT
        ]
        assert len(ep_nodes) == 5  # 2 + 2 + 1

    def test_has_parameter_nodes(self, graph: nx.DiGraph):
        param_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.PARAMETER
        ]
        assert len(param_nodes) > 0

    def test_has_response_field_nodes(self, graph: nx.DiGraph):
        rf_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.RESPONSE_FIELD
        ]
        assert len(rf_nodes) > 0

    def test_has_concept_nodes(self, graph: nx.DiGraph):
        concept_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.CONCEPT
        ]
        # All 3 tools share "travel" category → 1 concept node
        assert len(concept_nodes) == 1

    def test_has_endpoint_edges(self, graph: nx.DiGraph):
        ep_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.HAS_ENDPOINT
        ]
        assert len(ep_edges) == 5

    def test_has_tagged_with_edges(self, graph: nx.DiGraph):
        tagged = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.TAGGED_WITH
        ]
        assert len(tagged) == 3

    def test_has_compatible_with_edges(self, graph: nx.DiGraph):
        compat = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.COMPATIBLE_WITH
        ]
        # weather_api endpoints return 'city'; maps_api + booking_api require 'city'
        assert len(compat) > 0

    def test_endpoint_node_has_tool_id(self, graph: nx.DiGraph):
        ep_nid = node_id(NodeType.ENDPOINT, "get_forecast", parent="weather_api")
        assert graph.has_node(ep_nid)
        assert graph.nodes[ep_nid]["tool_id"] == "weather_api"

    def test_five_node_types_present(self, graph: nx.DiGraph):
        types_present = {d.get("type") for _, d in graph.nodes(data=True)}
        expected = {
            NodeType.TOOL, NodeType.ENDPOINT, NodeType.PARAMETER,
            NodeType.RESPONSE_FIELD, NodeType.CONCEPT,
        }
        assert expected == types_present


# ---------------------------------------------------------------------------
# ToolChain model
# ---------------------------------------------------------------------------

class TestToolChain:
    def test_flat_steps_sequential(self):
        chain = ToolChain(
            steps=[["ep:a.x"], ["ep:b.y"], ["ep:c.z"]],
            pattern="multi_step",
            tool_ids=["a", "b", "c"],
        )
        assert chain.flat_steps == ["ep:a.x", "ep:b.y", "ep:c.z"]

    def test_flat_steps_parallel(self):
        chain = ToolChain(
            steps=[["ep:a.x", "ep:b.y", "ep:c.z"]],
            pattern="parallel",
            tool_ids=["a", "b", "c"],
        )
        assert chain.flat_steps == ["ep:a.x", "ep:b.y", "ep:c.z"]

    def test_flat_steps_hybrid(self):
        chain = ToolChain(
            steps=[["ep:a.x"], ["ep:b.y", "ep:c.z"]],
            pattern="hybrid",
            tool_ids=["a", "b", "c"],
        )
        assert chain.flat_steps == ["ep:a.x", "ep:b.y", "ep:c.z"]


# ---------------------------------------------------------------------------
# ToolChainSampler
# ---------------------------------------------------------------------------

class TestToolChainSampler:
    @pytest.fixture
    def sampler(self, graph: nx.DiGraph) -> ToolChainSampler:
        return ToolChainSampler(graph, seed=42)

    def test_multi_step_returns_chain(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="multi_step", min_steps=3)
        assert isinstance(chain, ToolChain)
        assert chain.pattern == "multi_step"

    def test_multi_step_min_steps(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="multi_step", min_steps=3)
        assert len(chain.flat_steps) >= 3

    def test_multi_step_min_two_tools(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="multi_step", min_steps=3)
        assert len(set(chain.tool_ids)) >= 2

    def test_parallel_returns_chain(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="parallel", min_steps=3)
        assert isinstance(chain, ToolChain)
        assert chain.pattern == "parallel"

    def test_parallel_single_step_group(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="parallel", min_steps=3)
        assert len(chain.steps) == 1
        assert len(chain.steps[0]) >= 3

    def test_parallel_min_two_tools(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="parallel", min_steps=3)
        assert len(set(chain.tool_ids)) >= 2

    def test_flat_steps_nonempty(self, sampler: ToolChainSampler):
        chain = sampler.sample(pattern="multi_step", min_steps=3)
        assert len(chain.flat_steps) > 0

    def test_all_flat_steps_are_endpoint_nodes(self, sampler: ToolChainSampler, graph: nx.DiGraph):
        chain = sampler.sample(pattern="multi_step", min_steps=3)
        for step in chain.flat_steps:
            assert graph.has_node(step)
            assert graph.nodes[step]["type"] == NodeType.ENDPOINT

    def test_invalid_pattern_raises(self, sampler: ToolChainSampler):
        with pytest.raises(ValueError, match="Unknown pattern"):
            sampler.sample(pattern="invalid_pattern")

    def test_min_steps_zero_raises(self, sampler: ToolChainSampler):
        with pytest.raises(ValueError, match="min_steps must be >= 1"):
            sampler.sample(pattern="multi_step", min_steps=0)

    def test_deterministic_with_same_seed(self, graph: nx.DiGraph):
        s1 = ToolChainSampler(graph, seed=99)
        s2 = ToolChainSampler(graph, seed=99)
        c1 = s1.sample(pattern="multi_step", min_steps=3)
        c2 = s2.sample(pattern="multi_step", min_steps=3)
        assert c1.flat_steps == c2.flat_steps
