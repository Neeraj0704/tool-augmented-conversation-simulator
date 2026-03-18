"""End-to-end test: build artifacts and generate a dataset of at least 50 samples."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import pytest

from tacs.graph.builder import ToolGraphBuilder
from tacs.graph.models import NodeType
from tacs.registry.loader import load_tools
from tacs.registry.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("output")

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/ directory not found — run `tacs build` first",
)


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.load(ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def graph(registry: ToolRegistry) -> nx.DiGraph:
    pkl = ARTIFACTS_DIR / "graph.pkl"
    with open(pkl, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def run_a_conversations() -> list[dict]:
    path = OUTPUT_DIR / "run_a.jsonl"
    if not path.exists():
        pytest.skip("run_a.jsonl not found — run `tacs generate --no-corpus-memory`")
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def run_b_conversations() -> list[dict]:
    path = OUTPUT_DIR / "run_b.jsonl"
    if not path.exists():
        pytest.skip("run_b.jsonl not found — run `tacs generate`")
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------

class TestBuildArtifacts:
    def test_registry_pkl_exists(self):
        assert (ARTIFACTS_DIR / "registry.pkl").exists()

    def test_graph_pkl_exists(self):
        assert (ARTIFACTS_DIR / "graph.pkl").exists()

    def test_build_meta_exists(self):
        assert (ARTIFACTS_DIR / "build_meta.json").exists()

    def test_build_meta_fields(self):
        meta = json.loads((ARTIFACTS_DIR / "build_meta.json").read_text())
        assert "tool_count" in meta
        assert "endpoint_count" in meta
        assert "node_count" in meta
        assert meta["tool_count"] > 0
        assert meta["endpoint_count"] > 0

    def test_registry_loads(self, registry: ToolRegistry):
        assert registry.tool_count > 0
        assert registry.endpoint_count > 0

    def test_graph_loads(self, graph: nx.DiGraph):
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_graph_has_required_node_types(self, graph: nx.DiGraph):
        """Tool, Endpoint, Parameter, Concept are always required.
        ResponseField is optional — only present when response_examples exist."""
        types = {d.get("type") for _, d in graph.nodes(data=True)}
        required = {NodeType.TOOL, NodeType.ENDPOINT, NodeType.PARAMETER, NodeType.CONCEPT}
        assert required.issubset(types)

    def test_graph_has_compatible_with_or_concept_edges(self, graph: nx.DiGraph):
        """COMPATIBLE_WITH edges require response_examples data (optional).
        At minimum the graph must have TAGGED_WITH edges for concept-based sampling."""
        from tacs.graph.models import EdgeType
        tagged = [
            e for e in graph.edges(data=True)
            if e[2].get("type") == EdgeType.TAGGED_WITH
        ]
        assert len(tagged) > 0


# ---------------------------------------------------------------------------
# Dataset tests (≥50 samples, all required fields present)
# ---------------------------------------------------------------------------

REQUIRED_META_FIELDS = {
    "seed",
    "tool_ids_used",
    "num_turns",
    "num_clarification_questions",
    "memory_grounding_rate",
    "corpus_memory_enabled",
}

REQUIRED_TOP_FIELDS = {
    "conversation_id",
    "messages",
    "tool_calls",
    "tool_outputs",
    "metadata",
}


def _validate_conversation(conv: dict, idx: int) -> list[str]:
    errors = []
    missing_top = REQUIRED_TOP_FIELDS - set(conv.keys())
    if missing_top:
        errors.append(f"[{idx}] missing top-level fields: {missing_top}")
        return errors

    missing_meta = REQUIRED_META_FIELDS - set(conv["metadata"].keys())
    if missing_meta:
        errors.append(f"[{idx}] missing metadata fields: {missing_meta}")

    roles = {m["role"] for m in conv["messages"]}
    if "user" not in roles:
        errors.append(f"[{idx}] no user message")
    if "assistant" not in roles:
        errors.append(f"[{idx}] no assistant message")

    for tc in conv["tool_calls"]:
        if "endpoint" not in tc or "arguments" not in tc or "step" not in tc:
            errors.append(f"[{idx}] malformed tool_call: {tc}")

    for to in conv["tool_outputs"]:
        if "endpoint" not in to or "output" not in to or "step" not in to:
            errors.append(f"[{idx}] malformed tool_output: {to}")

    return errors


class TestRunADataset:
    def test_has_at_least_50_conversations(self, run_a_conversations: list[dict]):
        assert len(run_a_conversations) >= 50

    def test_all_conversations_have_required_fields(self, run_a_conversations: list[dict]):
        all_errors = []
        for i, conv in enumerate(run_a_conversations):
            all_errors.extend(_validate_conversation(conv, i))
        assert all_errors == [], "\n".join(all_errors)

    def test_corpus_memory_disabled(self, run_a_conversations: list[dict]):
        for conv in run_a_conversations:
            assert conv["metadata"]["corpus_memory_enabled"] is False

    def test_multi_step_tool_calls(self, run_a_conversations: list[dict]):
        multi_step = sum(1 for c in run_a_conversations if len(c["tool_calls"]) >= 3)
        assert multi_step > 0, "No conversations with >= 3 tool calls"

    def test_multi_tool_usage(self, run_a_conversations: list[dict]):
        multi_tool = sum(
            1 for c in run_a_conversations
            if len(set(c["metadata"]["tool_ids_used"])) >= 2
        )
        assert multi_tool > 0, "No conversations with >= 2 distinct tools"

    def test_memory_grounding_rate_in_bounds(self, run_a_conversations: list[dict]):
        for conv in run_a_conversations:
            rate = conv["metadata"]["memory_grounding_rate"]
            if rate is not None:
                assert 0.0 <= rate <= 1.0

    def test_tool_outputs_match_tool_calls(self, run_a_conversations: list[dict]):
        for conv in run_a_conversations:
            assert len(conv["tool_calls"]) == len(conv["tool_outputs"])

    def test_unique_conversation_ids(self, run_a_conversations: list[dict]):
        ids = [c["conversation_id"] for c in run_a_conversations]
        assert len(ids) == len(set(ids))


class TestRunBDataset:
    def test_has_at_least_50_conversations(self, run_b_conversations: list[dict]):
        assert len(run_b_conversations) >= 50

    def test_all_conversations_have_required_fields(self, run_b_conversations: list[dict]):
        all_errors = []
        for i, conv in enumerate(run_b_conversations):
            all_errors.extend(_validate_conversation(conv, i))
        assert all_errors == [], "\n".join(all_errors)

    def test_corpus_memory_enabled(self, run_b_conversations: list[dict]):
        for conv in run_b_conversations:
            assert conv["metadata"]["corpus_memory_enabled"] is True

    def test_multi_step_tool_calls(self, run_b_conversations: list[dict]):
        multi_step = sum(1 for c in run_b_conversations if len(c["tool_calls"]) >= 3)
        assert multi_step > 0

    def test_multi_tool_usage(self, run_b_conversations: list[dict]):
        multi_tool = sum(
            1 for c in run_b_conversations
            if len(set(c["metadata"]["tool_ids_used"])) >= 2
        )
        assert multi_tool > 0
