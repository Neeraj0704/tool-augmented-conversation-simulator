from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from tacs.config import Config
from tacs.llm import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all TACS agents.

    Provides shared access to the LLM client and config.
    Subclasses must implement run() with their own logic
    and type-specific parameters.

    Does NOT contain any conversation logic, memory access,
    or prompt templates — those belong in each subclass.
    """

    def __init__(self, llm: LLMClient, config: Config) -> None:
        self._llm = llm
        self._config = config

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the agent's task. Must be implemented by each subclass."""
        raise NotImplementedError
