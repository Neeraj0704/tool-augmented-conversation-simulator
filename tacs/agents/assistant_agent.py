from __future__ import annotations

import json
import logging
from typing import Any

from tacs.agents.base import BaseAgent
from tacs.agents.models import (
    AssistantAction,
    ConversationPlan,
    Message,
    ToolCall,
)
from tacs.config import Config
from tacs.graph.sampler import ToolChain
from tacs.llm import LLMClient
from tacs.memory.store import MemoryStore
from tacs.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AssistantAgent(BaseAgent):
    """Decides assistant actions: clarify, tool_call, or final respond.

    Clarify vs tool_call is rule-based (not LLM-decided). The LLM is
    only used to write natural language content and fill tool arguments.
    Session memory is injected for non-first steps using the exact
    assessment prompt format.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: Config,
        memory: MemoryStore,
        registry: ToolRegistry,
    ) -> None:
        super().__init__(llm, config)
        self._memory = memory
        self._registry = registry

    def run(
        self,
        plan: ConversationPlan,
        history: list[Message],
        step: int,
        session_state: dict[str, Any],
        tool_chain: ToolChain,
        final: bool = False,
    ) -> AssistantAction:
        """Decide and execute the next assistant action.

        Phases:
          0 — final response (when final=True)
          1 — session memory read (step > 0 only)
          2 — rule-based clarify vs tool_call decision
          3a — generate clarifying question via LLM
          3b — fill tool arguments via LLM
        Falls back gracefully on any LLM or parse failure.
        """
        # Phase 0 — final response
        if final:
            return self._final_response(plan, history)

        # Resolve endpoint from tool chain node ID
        endpoint, tool_id, endpoint_name = self._resolve_endpoint(tool_chain, step)
        if endpoint is None:
            logger.warning(
                "AssistantAgent: could not resolve endpoint at step=%d", step
            )
            return self._fallback(step, session_state, endpoint_name or "unknown")

        # Phase 1 — session memory read
        if step > 0:
            entries = self._memory.search(
                query=endpoint_name,
                scope="session",
                top_k=self._config.memory_top_k,
            )
            grounded = len(entries) > 0
        else:
            entries = []
            grounded = False

        # Phase 2 — rule-based clarify vs tool_call
        required_params = [p for p in endpoint.parameters if p.required]
        history_text = " ".join(m.content for m in history if m.role == "user")
        missing = [
            p
            for p in required_params
            if p.name not in session_state
            and p.name.lower() not in history_text.lower()
        ]

        already_clarified = any(m.role == "assistant" for m in history)
        action = "clarify" if (step == 0 and missing and not already_clarified) else "tool_call"

        # Phase 3a — clarify
        if action == "clarify":
            return self._clarify(plan, missing)

        # Phase 3b — tool_call
        return self._tool_call(
            plan=plan,
            history_text=history_text,
            endpoint_name=endpoint_name,
            endpoint=endpoint,
            entries=entries,
            session_state=session_state,
            grounded=grounded,
            step=step,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_endpoint(self, tool_chain: ToolChain, step: int) -> tuple:
        """Parse node ID and return (endpoint, tool_id, endpoint_name).

        Node IDs have the form "endpoint:tool_id.endpoint_name".
        Returns (None, None, None) if the step index is out of range
        or the registry lookup fails.
        """
        try:
            ep_node_id = tool_chain.flat_steps[step]
            parts = ep_node_id.split(":", 1)[1]
            tool_id = parts.rsplit(".", 1)[0]
            endpoint_name = parts.rsplit(".", 1)[1]
            endpoint = self._registry.get_endpoint(tool_id, endpoint_name)
            return endpoint, tool_id, endpoint_name
        except Exception as exc:
            logger.warning("AssistantAgent: endpoint resolution failed: %s", exc)
            return None, None, None

    def _final_response(
        self, plan: ConversationPlan, history: list[Message]
    ) -> AssistantAction:
        """Generate a natural closing response after all tool calls."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Summarise what you have done and give the user a clear, "
                    "natural final answer. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The user wanted: {plan.scenario}\n"
                    f"Provide a helpful closing response."
                ),
            },
        ]
        try:
            content = self._llm.complete(messages).strip()
        except Exception as exc:
            logger.warning("AssistantAgent: final response LLM failed: %s", exc)
            content = "I've completed your request. Is there anything else I can help you with?"

        logger.info("AssistantAgent: action=respond")
        return AssistantAction(
            action="respond",
            message=Message(role="assistant", content=content),
            tool_call=None,
            grounded=False,
        )

    def _clarify(self, plan: ConversationPlan, missing: list) -> AssistantAction:
        """Ask the user for missing required parameters."""
        missing_names = ", ".join(p.name for p in missing)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Ask concise questions.",
            },
            {
                "role": "user",
                "content": (
                    f"You are an AI assistant helping a user who wants:\n"
                    f"{plan.scenario}\n\n"
                    f"To proceed you need: {missing_names}\n\n"
                    f"Ask the user for this information naturally. "
                    f"One question only. Be concise."
                ),
            },
        ]
        try:
            content = self._llm.complete(messages).strip()
        except Exception as exc:
            logger.warning("AssistantAgent: clarify LLM failed: %s", exc)
            content = f"Could you please provide your {missing_names}?"

        logger.info("AssistantAgent: action=clarify missing=%s", missing_names)
        return AssistantAction(
            action="clarify",
            message=Message(role="assistant", content=content),
            tool_call=None,
            grounded=False,
        )

    def _tool_call(
        self,
        plan: ConversationPlan,
        history_text: str,
        endpoint_name: str,
        endpoint,
        entries: list[dict],
        session_state: dict[str, Any],
        grounded: bool,
        step: int,
    ) -> AssistantAction:
        """Fill tool arguments and return a tool_call action."""
        schema_json = json.dumps(
            [p.model_dump() for p in endpoint.parameters], default=str
        )
        session_json = json.dumps(session_state, default=str)

        if entries:
            memory_text = "\n".join(e.get("memory", str(e)) for e in entries)
            prompt = (
                f"[Memory context]\n{memory_text}\n\n"
                f"Given the above context and the current tool schema, "
                f"fill in the arguments for {endpoint_name}.\n\n"
                f"Schema: {schema_json}\n"
                f"session_state: {session_json}\n\n"
                f'Return JSON only: {{"param_name": value}}'
            )
        else:
            prompt = (
                f"Fill in the arguments for {endpoint_name}.\n"
                f"Schema: {schema_json}\n"
                f"session_state: {session_json}\n"
                f"User said: {history_text[-500:]}\n\n"
                f'Return JSON only: {{"param_name": value}}'
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant filling tool call arguments. "
                    "Return valid JSON only — no markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._llm.complete(messages).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            parsed_args: dict[str, Any] = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "AssistantAgent: argument JSON parse failed step=%d: %s", step, exc
            )
            parsed_args = {}

        # session_state exact values always win
        for key, value in session_state.items():
            if any(p.name == key for p in endpoint.parameters):
                parsed_args[key] = value

        logger.info(
            "AssistantAgent: action=tool_call endpoint=%s grounded=%s args_keys=%s",
            endpoint_name,
            grounded,
            list(parsed_args.keys()),
        )
        return AssistantAction(
            action="tool_call",
            message=Message(
                role="assistant",
                content=f"Let me {endpoint_name} for you.",
            ),
            tool_call=ToolCall(
                endpoint=endpoint_name,
                arguments=parsed_args,
                step=step,
            ),
            grounded=grounded,
        )

    def _fallback(
        self,
        step: int,
        session_state: dict[str, Any],
        endpoint_name: str,
    ) -> AssistantAction:
        """Return a safe fallback action when endpoint resolution fails."""
        if step == 0:
            return AssistantAction(
                action="clarify",
                message=Message(
                    role="assistant",
                    content="Could you provide more details about what you need?",
                ),
                tool_call=None,
                grounded=False,
            )
        return AssistantAction(
            action="tool_call",
            message=Message(
                role="assistant",
                content=f"Let me {endpoint_name} for you.",
            ),
            tool_call=ToolCall(
                endpoint=endpoint_name,
                arguments=dict(session_state),
                step=step,
            ),
            grounded=False,
        )
