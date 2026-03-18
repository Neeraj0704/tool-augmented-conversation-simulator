from __future__ import annotations

import logging

from tacs.agents.base import BaseAgent
from tacs.agents.models import ConversationPlan, Message
from tacs.config import Config
from tacs.llm import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a user interacting with an AI assistant. "
    "Respond naturally and conversationally. "
    "Never mention tool names or API calls directly."
)


class UserProxyAgent(BaseAgent):
    """Simulates user utterances within a conversation.

    Generates the opening message from the scenario on the first turn,
    and responds to assistant clarifying questions on subsequent turns.
    """

    def __init__(self, llm: LLMClient, config: Config) -> None:
        super().__init__(llm, config)

    def run(self, plan: ConversationPlan, history: list[Message]) -> Message:
        """Generate the next user message.

        On the first turn (empty history), produces an opening message
        from the scenario. On follow-up turns, responds to the last
        assistant clarifying question. Falls back to a plain message
        derived from the scenario if the LLM call fails.
        """
        if not history:
            user_prompt = (
                f"You are a user who wants to: {plan.scenario}\n"
                f"Write your opening message to an AI assistant. "
                f"Be natural and conversational. Do not mention "
                f"tools directly. "
                f"Intentionally omit some specific details like "
                f"exact dates, locations, or preferences — this "
                f"will prompt the assistant to ask clarifying "
                f"questions before proceeding."
            )
        else:
            last_assistant = next(
                (m.content for m in reversed(history) if m.role == "assistant"),
                None,
            )
            if last_assistant is None:
                logger.warning("UserProxyAgent: no assistant message found in history")
                return Message(role="user", content=f"I need help with: {plan.scenario}")

            user_prompt = (
                f"You are a user who wants to: {plan.scenario}\n"
                f"The assistant just asked: {last_assistant}\n"
                f"Answer naturally and provide the missing information."
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            content = self._llm.complete(messages).strip()
            logger.info(
                "UserProxyAgent: turn=%d content_len=%d",
                len(history),
                len(content),
            )
            return Message(role="user", content=content)
        except Exception as exc:
            logger.warning("UserProxyAgent: LLM call failed: %s", exc)
            return Message(role="user", content=f"I need help with: {plan.scenario}")