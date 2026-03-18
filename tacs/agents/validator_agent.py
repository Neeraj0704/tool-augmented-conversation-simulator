from __future__ import annotations

import logging

from tacs.agents.base import BaseAgent
from tacs.agents.models import Conversation, ValidationResult
from tacs.config import Config
from tacs.llm import LLMClient

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """Validates a completed conversation against assessment hard requirements.

    Purely rule-based — no LLM calls. Checks tool call counts, distinct
    tool usage, message presence, clarification questions, grounding rate
    bounds, and required metadata fields.
    """

    def __init__(self, llm: LLMClient, config: Config) -> None:
        super().__init__(llm, config)

    def run(self, conversation: Conversation) -> ValidationResult:
        """Validate conversation against all hard requirements.

        Runs all checks in order, collecting errors. Returns a
        ValidationResult with valid=True only if no errors were found.
        """
        errors: list[str] = []

        # Check 1 — minimum tool calls
        num_tool_calls = len(conversation.tool_calls)
        if num_tool_calls < self._config.min_tool_calls:
            errors.append(
                f"Only {num_tool_calls} tool calls, "
                f"need >= {self._config.min_tool_calls}"
            )

        # Check 2 — minimum distinct tools (from metadata, not endpoints)
        distinct_tool_ids = len(set(conversation.metadata.tool_ids_used))
        if distinct_tool_ids < self._config.min_distinct_tools:
            errors.append(
                f"Only {distinct_tool_ids} distinct tools, "
                f"need >= {self._config.min_distinct_tools}"
            )

        # Check 3 — has messages
        if len(conversation.messages) == 0:
            errors.append("Conversation has no messages")

        # Check 4 — has clarification question (tracked, not an error)
        has_clarification = any(
            m.role == "assistant" and "?" in m.content
            for m in conversation.messages
        )

        # Check 5 — memory_grounding_rate in valid range
        rate = conversation.metadata.memory_grounding_rate
        if rate is not None and not (0.0 <= rate <= 1.0):
            errors.append(
                f"memory_grounding_rate {rate} out of range [0,1]"
            )

        # Check 6 — required metadata fields set
        if conversation.metadata.num_turns <= 0:
            errors.append("num_turns not set")
        if not conversation.metadata.tool_ids_used:
            errors.append("tool_ids_used is empty")

        # Check 7 — num_clarification_questions is valid
        if conversation.metadata.num_clarification_questions < 0:
            errors.append("num_clarification_questions is negative")

        # Check 8 — tool call steps are sequential
        if conversation.tool_calls:
            steps = [tc.step for tc in conversation.tool_calls]
            expected = list(range(len(steps)))
            if steps != expected:
                errors.append(
                    f"Tool call steps not sequential: {steps}"
                )

        valid = len(errors) == 0

        if valid:
            logger.info(
                "ValidatorAgent: PASS conversation_id=%s "
                "tool_calls=%d distinct_tools=%d has_clarification=%s",
                conversation.conversation_id,
                num_tool_calls,
                distinct_tool_ids,
                has_clarification,
            )
        else:
            logger.warning(
                "ValidatorAgent: FAIL conversation_id=%s errors=%s",
                conversation.conversation_id,
                errors,
            )

        return ValidationResult(
            valid=valid,
            errors=errors,
            num_tool_calls=num_tool_calls,
            num_distinct_tools=distinct_tool_ids,
            has_clarification=has_clarification,
            memory_grounding_rate=conversation.metadata.memory_grounding_rate,
        )
