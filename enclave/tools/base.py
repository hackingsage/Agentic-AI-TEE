"""Base tool class and tool registry.

Every tool must subclass BaseTool and implement run() and schema_xml().
Tool names must be snake_case and unique across the registry.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from enclave.agent.models import ToolOutput

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Attributes:
        name: Unique snake_case identifier for this tool.
        description: Human-readable description (used in the LLM system prompt).
    """

    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments.

        Returns:
            ToolOutput with success status, result, and optional error.
        """
        raise NotImplementedError

    @abstractmethod
    def tool_definition(self) -> dict[str, Any]:
        """Return the tool definition in Anthropic API format.

        Example:
        {
            "name": "code_exec",
            "description": "Execute code in a sandboxed environment...",
            "input_schema": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript", "bash"]},
                    "code": {"type": "string", "description": "The code to execute"},
                },
                "required": ["language", "code"]
            }
        }
        """
        raise NotImplementedError

    def validate_args(self, args: dict[str, Any]) -> str | None:
        """Validate tool arguments. Return error string if invalid, None if ok.

        Default implementation accepts any args. Override for strict validation.
        """
        return None


@dataclass
class ToolSpec:
    """Metadata about a registered tool."""

    name: str
    description: str
    tool: BaseTool


class ToolRegistry:
    """Collects and validates all tool instances.

    Enforces unique snake_case names and provides lookup by name.
    """

    _NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises ValueError on duplicate or invalid name."""
        if not self._NAME_PATTERN.match(tool.name):
            raise ValueError(
                f"Tool name '{tool.name}' must be snake_case "
                f"(matching pattern {self._NAME_PATTERN.pattern})"
            )
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: '{tool.name}'")

        self._tools[tool.name] = ToolSpec(
            name=tool.name,
            description=tool.description,
            tool=tool,
        )
        logger.info("tool_registered", extra={"tool": tool.name})

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name. Returns None if not found."""
        spec = self._tools.get(name)
        return spec.tool if spec else None

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools in registration order."""
        return [spec.tool for spec in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def build_tool_definitions(self) -> list[dict[str, Any]]:
        """Build the list of tool definitions for the LLM API."""
        return [spec.tool.tool_definition() for spec in self._tools.values()]
