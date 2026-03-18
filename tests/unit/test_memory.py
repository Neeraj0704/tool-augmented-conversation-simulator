"""Unit tests for MemoryStore: add→search round-trip and scope isolation."""
from __future__ import annotations

import pytest

from tacs.memory.store import MemoryStore


def _try_create_store() -> MemoryStore | None:
    """Attempt to create a MemoryStore; return None if the backend is unavailable."""
    try:
        return MemoryStore()
    except Exception:
        return None


_store_instance = _try_create_store()

pytestmark = pytest.mark.skipif(
    _store_instance is None,
    reason=(
        "MemoryStore could not be initialised — "
        "ensure the configured LLM backend is running "
        "(e.g. Ollama for TACS_LLM_BACKEND=ollama, or set OPENAI_API_KEY for openai)."
    ),
)


@pytest.fixture(scope="module")
def store() -> MemoryStore:
    """One MemoryStore instance shared across all tests in this module."""
    return _store_instance  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# add → search round-trip
# ---------------------------------------------------------------------------

class TestAddSearch:
    def test_add_then_search_returns_entry(self, store: MemoryStore):
        store.add(
            content="Flight F99 from NYC to Tokyo confirmed",
            scope="session",
            metadata={"conversation_id": "test_001", "step": 0, "endpoint": "flight_search"},
        )
        results = store.search(query="flight NYC Tokyo", scope="session", top_k=5)
        assert len(results) > 0

    def test_search_returns_list_of_dicts(self, store: MemoryStore):
        store.add(
            content="Hotel booking confirmed in Tokyo",
            scope="session",
            metadata={"conversation_id": "test_001", "step": 1, "endpoint": "hotel_search"},
        )
        results = store.search(query="hotel Tokyo", scope="session", top_k=5)
        assert isinstance(results, list)
        for entry in results:
            assert isinstance(entry, dict)

    def test_search_top_k_limits_results(self, store: MemoryStore):
        for i in range(5):
            store.add(
                content=f"Tool output step {i}: result_{i}",
                scope="session",
                metadata={"conversation_id": "test_002", "step": i, "endpoint": f"tool_{i}"},
            )
        results = store.search(query="Tool output step result", scope="session", top_k=2)
        assert len(results) <= 2

    def test_search_missing_scope_returns_empty(self, store: MemoryStore):
        # A scope that has never had anything added should return empty
        results = store.search(query="anything", scope="nonexistent_scope_xyz", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------

class TestScopeIsolation:
    def test_session_entries_not_in_corpus(self, store: MemoryStore):
        unique_marker = "UNIQUE_SESSION_MARKER_12345"
        store.add(
            content=unique_marker,
            scope="session",
            metadata={"conversation_id": "iso_test", "step": 0, "endpoint": "ep"},
        )
        # Querying corpus scope should NOT return session entries
        corpus_results = store.search(query=unique_marker, scope="corpus", top_k=5)
        contents = [r.get("memory", "") for r in corpus_results]
        assert not any(unique_marker in c for c in contents)

    def test_corpus_entries_not_in_session(self, store: MemoryStore):
        unique_marker = "UNIQUE_CORPUS_MARKER_67890"
        store.add(
            content=unique_marker,
            scope="corpus",
            metadata={"conversation_id": "iso_test", "tools": ["tool_a"], "pattern_type": "multi_step"},
        )
        # Querying session scope should NOT return corpus entries
        session_results = store.search(query=unique_marker, scope="session", top_k=5)
        contents = [r.get("memory", "") for r in session_results]
        assert not any(unique_marker in c for c in contents)

    def test_corpus_entry_found_in_corpus(self, store: MemoryStore):
        unique_marker = "CORPUS_ROUNDTRIP_TEST_99999"
        store.add(
            content=unique_marker,
            scope="corpus",
            metadata={"conversation_id": "iso_test2", "tools": ["tool_b"], "pattern_type": "parallel"},
        )
        results = store.search(query=unique_marker, scope="corpus", top_k=5)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# clear_scope
# ---------------------------------------------------------------------------

class TestClearScope:
    def test_clear_scope_removes_session_entries(self, store: MemoryStore):
        unique_marker = "CLEAR_TEST_MARKER_11111"
        store.add(
            content=unique_marker,
            scope="session",
            metadata={"conversation_id": "clear_test", "step": 0, "endpoint": "ep"},
        )
        # Confirm it's there
        before = store.search(query=unique_marker, scope="session", top_k=5)
        assert len(before) > 0

        store.clear_scope("session")

        after = store.search(query=unique_marker, scope="session", top_k=5)
        assert len(after) == 0

    def test_clear_session_does_not_affect_corpus(self, store: MemoryStore):
        unique_marker = "CORPUS_PERSIST_MARKER_22222"
        store.add(
            content=unique_marker,
            scope="corpus",
            metadata={"conversation_id": "persist_test", "tools": ["t"], "pattern_type": "multi_step"},
        )
        store.clear_scope("session")

        results = store.search(query=unique_marker, scope="corpus", top_k=5)
        assert len(results) > 0
