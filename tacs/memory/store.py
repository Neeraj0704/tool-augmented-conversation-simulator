from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MemoryStore:
    """mem0-backed memory store with strict scope isolation.

    Supports two scopes:
      - "session": tool outputs within one conversation
      - "corpus": summaries across all conversations

    Scope isolation is enforced natively via mem0's user_id parameter —
    session and corpus entries are stored under separate user_id namespaces.
    Does NOT handle retries, embedding config, or cross-scope queries.
    """

    def __init__(self) -> None:
        from mem0 import Memory
        from tacs.config import config as tacs_config

        backend = tacs_config.llm_backend
        if backend == "anthropic" and not tacs_config.openai_api_key:
            raise ValueError(
                "TACS_LLM_BACKEND=anthropic requires OPENAI_API_KEY to be set "
                "because Anthropic has no embedding API — OpenAI embeddings are "
                "used for the memory store. Please add OPENAI_API_KEY to your .env."
            )

        # Use path=":memory:" so QdrantClient runs in pure in-memory mode.
        # The alternative (path=None / default) creates a file-based Qdrant
        # whose CollectionPersistence opens a SQLite connection with
        # check_same_thread=True (the macOS default for THREADSAFE=2), which
        # then raises when mem0's worker threads call back into Qdrant.
        #
        # Backend-aware config:
        #   ollama    → Ollama LLM + Ollama embedder (nomic-embed-text, 768 dims)
        #   openai    → OpenAI LLM + OpenAI embedder (text-embedding-3-small, 1536 dims)
        #   anthropic → Anthropic LLM + OpenAI embedder (1536 dims)
        #               Anthropic has no embedding API so OpenAI embeddings are used;
        #               OPENAI_API_KEY must be set alongside ANTHROPIC_API_KEY.

        if backend == "openai":
            mem_config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": tacs_config.openai_model,
                        "api_key": tacs_config.openai_api_key,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": tacs_config.openai_api_key,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "tacs_memory",
                        "embedding_model_dims": 1536,
                        "path": ":memory:",
                    },
                },
            }
        elif backend == "anthropic":
            # Anthropic provides no embedding model — use OpenAI embeddings.
            # Requires OPENAI_API_KEY in addition to ANTHROPIC_API_KEY.
            mem_config = {
                "llm": {
                    "provider": "anthropic",
                    "config": {
                        "model": tacs_config.anthropic_model,
                        "api_key": tacs_config.anthropic_api_key,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        "api_key": tacs_config.openai_api_key,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "tacs_memory",
                        "embedding_model_dims": 1536,
                        "path": ":memory:",
                    },
                },
            }
        else:
            # Default: ollama (nomic-embed-text → 768 dims)
            mem_config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": tacs_config.ollama_model,
                        "ollama_base_url": tacs_config.ollama_base_url,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": tacs_config.ollama_embed_model,
                        "ollama_base_url": tacs_config.ollama_base_url,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "tacs_memory",
                        "embedding_model_dims": tacs_config.ollama_embed_dims,
                        "path": ":memory:",
                    },
                },
            }

        self._memory = Memory.from_config(mem_config)

    def add(self, content: str, scope: str, metadata: dict) -> None:
        """Store an entry under the given scope.

        Uses scope as user_id for native mem0 isolation. Stores content
        as-is (infer=False) — no LLM extraction. Never raises — logs
        and returns on failure so the pipeline is not interrupted.
        """
        try:
            self._memory.add(
                content,
                user_id=scope,
                metadata=metadata,
                infer=False,
            )
            logger.debug(
                "MemoryStore.add: scope=%s metadata_keys=%s",
                scope,
                list(metadata.keys()),
            )
        except Exception as exc:
            logger.warning(
                "MemoryStore.add failed: scope=%s error=%s", scope, exc
            )

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        """Return top_k entries matching query within the given scope only.

        Scope isolation is guaranteed by querying only under the scope's
        user_id namespace. Returns empty list on failure so callers can
        proceed without memory context.

        A non-empty return value counts as a grounded retrieval for
        memory_grounding_rate calculation — no score threshold applied.
        """
        try:
            result = self._memory.search(query, user_id=scope, limit=top_k)
            entries = result.get("results", [])
            logger.debug(
                "MemoryStore.search: scope=%s query_len=%d returned=%d",
                scope,
                len(query),
                len(entries),
            )
            return entries
        except Exception as exc:
            logger.warning(
                "MemoryStore.search failed: scope=%s error=%s", scope, exc
            )
            return []

    def clear_scope(self, scope: str) -> None:
        """Delete all entries for the given scope.

        NOTE: This method is NOT part of the assessment interface.
        It is a pipeline utility — called between conversations to
        reset session memory without touching corpus memory.

        Usage: memory.clear_scope("session") after each conversation.
        """
        try:
            self._memory.delete_all(user_id=scope)
            logger.debug("MemoryStore.clear_scope: scope=%s", scope)
        except Exception as exc:
            logger.warning(
                "MemoryStore.clear_scope failed: scope=%s error=%s", scope, exc
            )
