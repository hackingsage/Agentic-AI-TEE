"""Planner — builds system prompts and gets tool definitions.

No longer handles XML construction or parsing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from enclave.agent.llm_client import LLMClient
from enclave.agent.models import LLMResponse, Message
from enclave.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are Enclave Agent, an autonomous AI assistant running inside a Trusted Execution Environment (TEE).
Your task is: {task_description}

You have access to tools that you can call to accomplish the task. Use them as needed.

Rules:
- Think step by step before each action.
- Never fabricate tool outputs. If a tool fails, report the error and decide whether to retry or abort.
- When you have completed the task, provide your final response as plain text (do not call any more tools).
- The task is complete when you respond with only text and no tool calls.
"""


class Planner:
    """Orchestrates LLM interactions using native tool calling.

    Builds system prompts and passes tool definitions to the LLM Client.
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        self._llm = llm_client
        self._registry = tool_registry

    def build_system_prompt(self, task_description: str) -> str:
        """Build the complete system prompt."""
        return SYSTEM_PROMPT_TEMPLATE.format(task_description=task_description)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get the JSON schema tool definitions for the registered tools."""
        return self._registry.build_tool_definitions()

    async def plan_next_step(
        self,
        system_prompt: str,
        messages: list[Message],
        *,
        task_id: str = "",
        step_number: int = 0,
        max_tokens: int = 4096,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        """Call the LLM to plan the next step, passing the native tool definitions."""
        return await self._llm.call(
            system=system_prompt,
            messages=messages,
            tools=self.get_tool_definitions(),
            max_tokens=max_tokens,
            task_id=task_id,
            step_number=step_number,
            on_chunk=on_chunk,
        )
