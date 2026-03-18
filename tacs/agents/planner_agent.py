from __future__ import annotations

import json
import logging

from tacs.agents.base import BaseAgent
from tacs.agents.models import ConversationPlan
from tacs.config import Config
from tacs.graph.sampler import ToolChain
from tacs.llm import LLMClient
from tacs.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_FALLBACK_DOMAINS = {
    "weather": "weather",
    "flight": "travel",
    "booking": "travel",
    "hotel": "travel",
    "map": "navigation",
    "finance": "finance",
    "stock": "finance",
    "news": "news",
    "sport": "sports",
    "music": "entertainment",
    "movie": "entertainment",
    "restaurant": "food",
    "food": "food",
}

_SYSTEM_PROMPT = """\
You are a conversation scenario planner. Given a tool chain, create a realistic
user scenario that would naturally require those exact tools. Respond with valid
JSON only — no markdown, no explanation.

Response format:
{
  "scenario": "<one sentence describing what the user wants to accomplish>",
  "domain": "<single domain word, e.g. travel, finance, weather, food>",
  "pattern_type": "<e.g. sequential multi-step, parallel, hybrid>"
}"""


class PlannerAgent(BaseAgent):
    """Plans the conversation scenario for a given tool chain.

    Reads corpus memory (when enabled) to generate diverse plans that
    avoid repeating scenarios already seen. Falls back to a derived
    plan if the LLM call fails or returns invalid JSON.
    """

    def __init__(self, llm: LLMClient, config: Config, memory: MemoryStore) -> None:
        super().__init__(llm, config)
        self._memory = memory

    def run(
        self,
        tool_chain: ToolChain,
        corpus_memory_enabled: bool = True,
    ) -> ConversationPlan:
        """Plan a conversation scenario for the given tool chain.

        Queries corpus memory when enabled, injects prior summaries into
        the prompt, and asks the LLM for a JSON scenario plan. Returns a
        derived fallback plan if the LLM fails or JSON is invalid.
        """
        tool_chain_str = self._format_tool_chain(tool_chain)

        # Step 1 — read corpus memory if enabled
        corpus_entries: list[dict] = []
        if corpus_memory_enabled:
            corpus_entries = self._memory.search(
                query=tool_chain_str,
                scope="corpus",
                top_k=self._config.memory_top_k,
            )

        # Step 2 — build prompt (exact assessment format)
        user_prompt = self._build_prompt(tool_chain_str, corpus_entries)

        # Step 3 — call LLM, parse JSON
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self._llm.complete(messages)
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("```")[1]
                if raw_clean.startswith("json"):
                    raw_clean = raw_clean[4:]
            raw_clean = raw_clean.strip()
            plan_dict = json.loads(raw_clean)
            scenario = str(plan_dict["scenario"])
            domain = str(plan_dict["domain"])
            pattern_type = str(plan_dict["pattern_type"])
            logger.info(
                "PlannerAgent: domain=%s pattern=%s tools=%s",
                domain,
                pattern_type,
                tool_chain.tool_ids,
            )
            return ConversationPlan(
                scenario=scenario,
                domain=domain,
                pattern_type=pattern_type,
                tool_chain=tool_chain,
            )
        except Exception as exc:
            logger.warning("PlannerAgent: LLM call or JSON parse failed: %s", exc)
            return self._fallback_plan(tool_chain)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, tool_chain_str: str, corpus_entries: list[dict]) -> str:
        """Build the planning prompt using the exact assessment format."""
        if corpus_entries:
            summaries = "\n".join(
                e.get("memory", e.get("content", str(e))) for e in corpus_entries
            )
            return (
                f"[Prior conversations in corpus]\n{summaries}\n\n"
                f"Given the above, plan a new diverse conversation using the "
                f"following tool chain:\n{tool_chain_str}"
            )
        return (
            f"Plan a conversation using the following tool chain:\n{tool_chain_str}"
        )

    def _format_tool_chain(self, tool_chain: ToolChain) -> str:
        """Produce a human-readable tool chain description for prompts."""
        steps_str = " → ".join(
            f"[{', '.join(step)}]" for step in tool_chain.steps
        )
        return (
            f"Pattern: {tool_chain.pattern}\n"
            f"Tools: {', '.join(tool_chain.tool_ids)}\n"
            f"Steps: {steps_str}"
        )

    def _fallback_plan(self, tool_chain: ToolChain) -> ConversationPlan:
        """Derive a minimal plan from tool_chain without any LLM call."""
        domain = "general"
        for tool_id in tool_chain.tool_ids:
            key = tool_id.lower()
            for keyword, mapped in _FALLBACK_DOMAINS.items():
                if keyword in key:
                    domain = mapped
                    break
            if domain != "general":
                break

        first_tool = tool_chain.tool_ids[0] if tool_chain.tool_ids else "a tool"
        scenario = f"User wants to accomplish a task using {first_tool}."
        pattern_type = tool_chain.pattern.replace("_", " ")

        logger.info(
            "PlannerAgent: fallback plan domain=%s pattern=%s", domain, pattern_type
        )
        return ConversationPlan(
            scenario=scenario,
            domain=domain,
            pattern_type=pattern_type,
            tool_chain=tool_chain,
        )
