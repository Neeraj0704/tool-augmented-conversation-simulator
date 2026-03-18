from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """A single message in a conversation turn."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "tool", "system"]
    content: str


class ToolCall(BaseModel):
    """A tool call made by the assistant during a conversation."""

    model_config = ConfigDict(extra="ignore")

    endpoint: str
    arguments: dict[str, Any]
    step: int


class ToolOutput(BaseModel):
    """The mock response returned after executing a tool call."""

    model_config = ConfigDict(extra="ignore")

    endpoint: str
    output: dict[str, Any]
    step: int


class ConversationMetadata(BaseModel):
    """Metadata about a generated conversation for evaluation and analysis."""

    model_config = ConfigDict(extra="ignore")

    seed: int
    tool_ids_used: list[str]
    num_turns: int
    num_clarification_questions: int
    memory_grounding_rate: float | None
    corpus_memory_enabled: bool


class Conversation(BaseModel):
    """A complete synthetic conversation with tool calls, outputs, and metadata."""

    model_config = ConfigDict(extra="ignore")

    conversation_id: str
    messages: list[Message] = []
    tool_calls: list[ToolCall] = []
    tool_outputs: list[ToolOutput] = []
    metadata: ConversationMetadata
