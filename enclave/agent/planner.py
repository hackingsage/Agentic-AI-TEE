"""Planner — builds system prompts and parses LLM responses for tool calls.

Handles XML construction for the system prompt and XML parsing of
<tool_call> and <task_complete> blocks from LLM responses.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from enclave.agent.llm_client import LLMClient
from enclave.agent.models import (
    LLMResponse,
    Message,
    TaskComplete,
    ToolCall,
)
from enclave.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """<system>
You are Enclave Agent, an autonomous AI assistant running inside a Trusted Execution Environment.
Your task is: {task_description}

You have access to the following tools. To use a tool, respond with a <tool_call> block:
<tool_call>
  <name>tool_name</name>
  <args>
    <arg_name>value</arg_name>
  </args>
</tool_call>

Available tools:
{tool_schema_xml}

Rules:
- Think step by step before each tool call. Output your reasoning in <thinking> tags.
- Never fabricate tool outputs. If a tool fails, report the error and decide whether to retry or abort.
- Always output your actual final detailed response to the user OUTSIDE of any XML tags, so the user can read it.
- After completing the task and writing your final response, output a <task_complete> block with a short summary of the completed actions.
- The user cannot see your thinking, only your final response and tool outputs.
</system>"""


class Planner:
    """Orchestrates LLM interactions: prompt construction and response parsing.

    Builds XML system prompts from the tool registry and parses
    structured <tool_call> and <task_complete> blocks from responses.
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        self._llm = llm_client
        self._registry = tool_registry

    def build_system_prompt(self, task_description: str) -> str:
        """Build the complete system prompt with tool schemas."""
        tool_schema_xml = self._registry.build_schema_xml()
        return SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description,
            tool_schema_xml=tool_schema_xml,
        )

    async def plan_next_step(
        self,
        system_prompt: str,
        messages: list[Message],
        *,
        task_id: str = "",
        step_number: int = 0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the LLM to plan the next step."""
        return await self._llm.call(
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            task_id=task_id,
            step_number=step_number,
        )

    @staticmethod
    def parse_tool_calls(response_text: str) -> list[ToolCall]:
        """Parse <tool_call> blocks from the LLM response.

        Returns a list of ToolCall objects. Handles malformed XML gracefully.
        """
        tool_calls: list[ToolCall] = []

        # Find all <tool_call>...</tool_call> blocks
        pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>",
            re.DOTALL,
        )

        for match in pattern.finditer(response_text):
            xml_content = match.group(1).strip()
            try:
                tool_call = _parse_single_tool_call(xml_content)
                if tool_call:
                    tool_calls.append(tool_call)
            except Exception as exc:
                logger.warning(
                    "tool_call_parse_error",
                    extra={"error": str(exc), "xml_snippet": xml_content[:200]},
                )

        return tool_calls

    @staticmethod
    def parse_task_complete(response_text: str) -> TaskComplete | None:
        """Parse <task_complete> block from the LLM response.

        Returns TaskComplete if found, None otherwise.
        """
        pattern = re.compile(
            r"<task_complete>(.*?)</task_complete>",
            re.DOTALL,
        )

        match = pattern.search(response_text)
        if not match:
            return None

        xml_content = match.group(1).strip()
        try:
            # Try XML parsing first
            wrapped = f"<root>{xml_content}</root>"
            root = ET.fromstring(wrapped)
            summary_el = root.find("summary")
            if summary_el is not None and summary_el.text:
                return TaskComplete(summary=summary_el.text.strip())
        except ET.ParseError:
            pass

        # Fallback: treat the entire content as the summary
        if xml_content:
            return TaskComplete(summary=xml_content.strip())

        return TaskComplete(summary="Task completed.")

    @staticmethod
    def format_tool_outputs_xml(tool_name: str, output: Any) -> str:
        """Format tool output as XML for the conversation history."""
        success = output.success if hasattr(output, "success") else True
        result = output.result if hasattr(output, "result") else str(output)
        error = output.error if hasattr(output, "error") else None

        parts = [f"<tool_result>", f"  <name>{tool_name}</name>"]
        parts.append(f"  <success>{str(success).lower()}</success>")

        if result is not None:
            result_str = str(result)
            # Escape XML special characters
            result_str = (
                result_str.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(f"  <output>{result_str}</output>")

        if error:
            error_str = (
                str(error)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(f"  <error>{error_str}</error>")

        parts.append("</tool_result>")
        return "\n".join(parts)


def _parse_single_tool_call(xml_content: str) -> ToolCall | None:
    """Parse a single tool call from its inner XML content.

    Expected format:
        <name>tool_name</name>
        <args>
            <arg_name>value</arg_name>
        </args>
    """
    # Wrap in root element for valid XML
    wrapped = f"<root>{xml_content}</root>"

    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # Try to fix common LLM XML mistakes
        cleaned = xml_content.replace("&", "&amp;")
        wrapped = f"<root>{cleaned}</root>"
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError:
            logger.warning("unparseable_tool_call_xml")
            return None

    # Extract tool name
    name_el = root.find("name")
    if name_el is None or not name_el.text:
        logger.warning("tool_call_missing_name")
        return None

    tool_name = name_el.text.strip()

    # Validate tool name (snake_case, no special chars)
    if not re.match(r"^[a-z][a-z0-9_]*$", tool_name):
        logger.warning(
            "tool_call_invalid_name",
            extra={"name": tool_name},
        )
        return None

    # Extract arguments
    args: dict[str, Any] = {}
    args_el = root.find("args")
    if args_el is not None:
        for child in args_el:
            tag = child.tag
            text = child.text or ""
            text = text.strip()

            # Try to parse as JSON-like values
            if text.lower() == "true":
                args[tag] = True
            elif text.lower() == "false":
                args[tag] = False
            elif text.isdigit():
                args[tag] = int(text)
            else:
                try:
                    args[tag] = float(text)
                except ValueError:
                    args[tag] = text

    return ToolCall(name=tool_name, args=args)
