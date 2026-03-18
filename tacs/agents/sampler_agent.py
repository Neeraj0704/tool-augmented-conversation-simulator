from __future__ import annotations

import logging
import random

import networkx as nx

from tacs.agents.base import BaseAgent
from tacs.config import Config
from tacs.graph.sampler import ToolChain, ToolChainSampler
from tacs.llm import LLMClient

logger = logging.getLogger(__name__)

_PATTERNS = ["multi_step", "parallel", "hybrid"]


class SamplerAgent(BaseAgent):
    """Proposes a tool chain from the Tool Graph.

    Wraps ToolChainSampler in the agent interface. Always samples
    from the graph — never returns hardcoded tool lists.
    """

    def __init__(self, llm: LLMClient, config: Config, graph: nx.DiGraph) -> None:
        super().__init__(llm, config)
        self._graph = graph
        self._rng = random.Random(config.seed)

    def run(self, pattern: str | None = None) -> ToolChain:
        """Sample a tool chain from the graph.

        If pattern is None, one is chosen randomly. Tries remaining
        patterns in order if the chosen one fails.
        """
        # Hard requirement: tool chains MUST come from
        # the graph sampler — never hardcoded lists
        sampler = ToolChainSampler(self._graph, seed=self._config.seed)

        patterns_to_try: list[str] = []
        if pattern is not None:
            patterns_to_try = [pattern] + [p for p in _PATTERNS if p != pattern]
        else:
            shuffled = list(_PATTERNS)
            self._rng.shuffle(shuffled)
            patterns_to_try = shuffled

        last_error: Exception | None = None
        for p in patterns_to_try:
            try:
                chain = sampler.sample(p, min_steps=self._config.min_tool_calls)
                logger.info(
                    "SamplerAgent: pattern=%s steps=%d tools=%d",
                    chain.pattern,
                    len(chain.steps),
                    len(chain.tool_ids),
                )
                return chain
            except ValueError as exc:
                logger.warning("SamplerAgent: pattern=%s failed: %s", p, exc)
                last_error = exc

        raise ValueError(
            f"SamplerAgent failed all patterns {patterns_to_try}: {last_error}"
        )
