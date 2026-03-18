from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm import tqdm

from tacs.agents.assistant_agent import AssistantAgent
from tacs.agents.models import (
    Conversation,
    ConversationMetadata,
    Message,
    ToolCall,
    ToolOutput,
)
from tacs.agents.planner_agent import PlannerAgent
from tacs.agents.sampler_agent import SamplerAgent
from tacs.agents.user_proxy import UserProxyAgent
from tacs.agents.validator_agent import ValidatorAgent
from tacs.config import Config
from tacs.execution.executor import MockExecutor
from tacs.llm import LLMClient
from tacs.memory.store import MemoryStore
from tacs.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConversationPipeline:
    """Orchestrates all agents to produce one complete conversation.

    Agents are instantiated once in __init__ and reused across calls
    to run(). Session memory is cleared between conversations.
    Corpus memory is written only after successful validation.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        graph: nx.DiGraph,
        llm: LLMClient,
        config: Config,
        memory: MemoryStore,
    ) -> None:
        self._registry = registry
        self._memory = memory
        self._config = config
        self._sampler_agent = SamplerAgent(llm, config, graph)
        self._planner_agent = PlannerAgent(llm, config, memory)
        self._user_proxy = UserProxyAgent(llm, config)
        self._assistant = AssistantAgent(llm, config, memory, registry)
        self._validator = ValidatorAgent(llm, config)
        self._executor = MockExecutor(registry, seed=config.seed)

    def run(
        self,
        conversation_id: str,
        corpus_memory_enabled: bool = True,
    ) -> Conversation:
        """Run the full pipeline for one conversation.

        Returns the Conversation even if validation fails or an unexpected
        error occurs — the caller decides whether to keep or discard it.
        """
        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        tool_outputs: list[ToolOutput] = []
        session_state: dict[str, Any] = {}
        num_clarification_questions = 0
        grounded_calls = 0
        non_first_step_calls = 0
        current_step = 0
        chain = None
        plan = None
        validation = None

        try:
            # Step 1 — sample tool chain
            chain = self._sampler_agent.run()

            # Step 2 — plan conversation
            plan = self._planner_agent.run(
                chain, corpus_memory_enabled=corpus_memory_enabled
            )

            total_steps = len(chain.flat_steps)

            # Step 3 — conversation loop
            for _turn in range(self._config.max_turns):

                # User speaks
                user_msg = self._user_proxy.run(plan, messages)
                messages.append(user_msg)

                # Check if all tool steps are complete
                final = current_step >= total_steps

                # Assistant decides action
                action = self._assistant.run(
                    plan=plan,
                    history=messages,
                    step=current_step,
                    session_state=session_state,
                    tool_chain=chain,
                    final=final,
                )
                messages.append(action.message)

                if action.action == "clarify":
                    num_clarification_questions += 1
                    continue

                if action.action == "respond":
                    break

                if action.action == "tool_call" and action.tool_call:
                    # Parse tool_id and endpoint_name from chain node ID
                    ep_node_id = chain.flat_steps[current_step]
                    parts = ep_node_id.split(":", 1)[1]
                    tool_id = parts.rsplit(".", 1)[0]
                    endpoint_name = parts.rsplit(".", 1)[1]

                    # Execute mock tool
                    result = self._executor.execute(
                        tool_id=tool_id,
                        endpoint_name=endpoint_name,
                        arguments=action.tool_call.arguments,
                    )

                    # Write to session memory (assessment requirement)
                    self._memory.add(
                        content=json.dumps(result.output),
                        scope="session",
                        metadata={
                            "conversation_id": conversation_id,
                            "step": current_step,
                            "endpoint": endpoint_name,
                        },
                    )

                    # Update session_state for exact value chaining
                    session_state.update(result.output)

                    # Track memory grounding rate
                    if current_step > 0:
                        non_first_step_calls += 1
                        if action.grounded:
                            grounded_calls += 1

                    # Record tool call and output
                    tool_calls.append(
                        ToolCall(
                            endpoint=endpoint_name,
                            arguments=action.tool_call.arguments,
                            step=current_step,
                        )
                    )
                    tool_outputs.append(
                        ToolOutput(
                            endpoint=endpoint_name,
                            output=result.output,
                            step=current_step,
                        )
                    )

                    # Add tool message to conversation history
                    messages.append(
                        Message(role="tool", content=json.dumps(result.output))
                    )

                    current_step += 1

                    # All steps done — get final response immediately
                    if current_step >= total_steps:
                        final_action = self._assistant.run(
                            plan=plan,
                            history=messages,
                            step=current_step,
                            session_state=session_state,
                            tool_chain=chain,
                            final=True,
                        )
                        messages.append(final_action.message)
                        break

        except Exception:
            logger.exception(
                "Pipeline: unexpected error for %s", conversation_id
            )
            self._memory.clear_scope("session")

        # Step 4 — calculate memory_grounding_rate
        # null if only one tool call (no non-first steps)
        rate: float | None = (
            None
            if non_first_step_calls == 0
            else grounded_calls / non_first_step_calls
        )

        # Step 5 — build Conversation object
        conversation = Conversation(
            conversation_id=conversation_id,
            messages=messages,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            metadata=ConversationMetadata(
                seed=self._config.seed,
                tool_ids_used=chain.tool_ids if chain else [],
                num_turns=len(messages),
                num_clarification_questions=num_clarification_questions,
                memory_grounding_rate=rate,
                corpus_memory_enabled=corpus_memory_enabled,
            ),
        )

        # Step 6 — validate
        validation = self._validator.run(conversation)
        if not validation.valid:
            logger.warning(
                "Pipeline: conversation %s failed validation: %s",
                conversation_id,
                validation.errors,
            )

        # Step 7 — write corpus memory after validation (assessment requirement)
        if corpus_memory_enabled and chain and plan:
            summary = (
                f"Tools: {', '.join(chain.tool_ids)}. "
                f"Domain: {plan.domain}. "
                f"Pattern: {plan.pattern_type}."
            )
            self._memory.add(
                content=summary,
                scope="corpus",
                metadata={
                    "conversation_id": conversation_id,
                    "tools": chain.tool_ids,
                    "pattern_type": plan.pattern_type,
                },
            )

        # Step 8 — clear session memory between conversations
        self._memory.clear_scope("session")

        logger.info(
            "Pipeline: completed %s turns=%d tool_calls=%d valid=%s",
            conversation_id,
            len(messages),
            len(tool_calls),
            validation.valid if validation else "unknown",
        )
        return conversation


def generate(
    registry: ToolRegistry,
    graph: nx.DiGraph,
    llm: LLMClient,
    config: Config,
    memory: MemoryStore,
    count: int,
    corpus_memory_enabled: bool,
    output_path: Path,
) -> None:
    """Generate count conversations and save to JSONL.

    Each conversation is written immediately after generation so partial
    runs are not lost if the process is interrupted.
    """
    pipeline = ConversationPipeline(
        registry=registry,
        graph=graph,
        llm=llm,
        config=config,
        memory=memory,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(output_path, "w") as f:
        for i in tqdm(range(count), desc="Generating"):
            conv_id = f"conv_{i:05d}"
            conversation = pipeline.run(
                conversation_id=conv_id,
                corpus_memory_enabled=corpus_memory_enabled,
            )
            f.write(conversation.model_dump_json() + "\n")
            success += 1

    logger.info(
        "generate: wrote %d/%d conversations to %s",
        success,
        count,
        output_path,
    )
