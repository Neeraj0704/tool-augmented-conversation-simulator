from __future__ import annotations

import logging
from typing import Any

from tacs.config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM interface supporting ollama, openai, and anthropic backends.

    Wraps three backends behind a single `complete()` method so the rest of
    the codebase never imports ollama, openai, or anthropic directly.

    Does NOT handle retries, streaming, token counting, or prompt templating —
    those are the caller's responsibility.
    """

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or config.llm_backend
        self._client = self._init_client()

    def _init_client(self) -> Any:
        if self.backend == "ollama":
            import ollama
            return ollama
        elif self.backend == "openai":
            from openai import OpenAI
            return OpenAI(api_key=config.openai_api_key)
        elif self.backend == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages and return the assistant reply as a string."""
        logger.debug("LLM call backend=%s messages=%d", self.backend, len(messages))

        if self.backend == "ollama":
            response = self._client.chat(
                model=config.ollama_model,
                messages=messages,
                **kwargs,
            )
            return response.message.content

        elif self.backend == "openai":
            response = self._client.chat.completions.create(
                model=config.openai_model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content

        elif self.backend == "anthropic":
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            non_system = [m for m in messages if m["role"] != "system"]
            response = self._client.messages.create(
                model=config.anthropic_model,
                max_tokens=kwargs.pop("max_tokens", config.llm_max_tokens),
                system=system,
                messages=non_system,
                **kwargs,
            )
            return response.content[0].text
