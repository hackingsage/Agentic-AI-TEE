"""Core data structures for the Enclave agent.

All inter-component data is passed as dataclasses — no raw dicts.
Uses structured content blocks matching the Anthropic Messages API format
for native tool_use support (Claude Code-style agentic loop).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Union


# ─── Content Blocks ─────────────────────────────────────────────────────────── #


@dataclass
class TextBlock:
    """A text content block in a message."""

    text: str = ""
    type: str = "text"


@dataclass
class ToolUseBlock:
    """A tool_use content block from the model's response.

    Represents a structured tool invocation with a unique ID,
    tool name, and JSON arguments — no XML parsing needed.
    """

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class ToolResultBlock:
    """A tool_result content block to feed back to the model.

    Matches the tool_use_id from the corresponding ToolUseBlock
    so the model knows which tool call this result belongs to.
    """

    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False
    type: str = "tool_result"


# Union type for all content block variants
ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


# ─── Messages ───────────────────────────────────────────────────────────────── #


@dataclass
class Message:
    """A single message in the conversation history.

    For simple user messages, content is a plain string.
    For assistant responses with tool calls, or user messages with tool results,
    content is a list of ContentBlock objects.
    """

    role: str  # "user" or "assistant"
    content: str | list[ContentBlock] = ""

    @property
    def text(self) -> str:
        """Extract text content from the message."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            b.text for b in self.content if isinstance(b, TextBlock) and b.text
        )


# ─── Tool Dispatch ──────────────────────────────────────────────────────────── #


@dataclass
class ToolCall:
    """A parsed tool invocation for internal dispatch by the ToolRouter.

    Constructed from ToolUseBlock content blocks.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    """Result of a tool execution."""

    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


# ─── LLM Response ──────────────────────────────────────────────────────────── #


@dataclass
class LLMResponse:
    """Response from the LLM provider with structured content blocks.

    Instead of a flat text string, the response contains a list of
    ContentBlock objects (TextBlock and/or ToolUseBlock), matching
    the native API response format.
    """

    content: list[ContentBlock] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    stop_reason: str = ""  # "end_turn", "tool_use", "max_tokens"

    @property
    def text(self) -> str:
        """Extract concatenated text from all TextBlocks."""
        parts = [b.text for b in self.content if isinstance(b, TextBlock) and b.text]
        return "\n".join(parts)

    @property
    def tool_use_blocks(self) -> list[ToolUseBlock]:
        """Extract all tool_use blocks from the response."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def has_tool_use(self) -> bool:
        """Check if the response contains any tool_use blocks."""
        return any(isinstance(b, ToolUseBlock) for b in self.content)

    @property
    def cost_usd(self) -> float:
        """Estimate cost using Claude Sonnet pricing ($3/1M input, $15/1M output)."""
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


# ─── Step & Task Results ────────────────────────────────────────────────────── #


@dataclass
class StepResult:
    """Result of a single agent step (LLM call + tool execution)."""

    step_number: int
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    output: ToolOutput | None = None
    llm_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


@dataclass
class TaskRequest:
    """Incoming task from the user."""

    description: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str = "anonymous"
    budget_usd: float = 5.0
    max_steps: int = 50
    timeout_seconds: float = 300.0
    tool_allowlist: list[str] | None = None  # None = all tools allowed
    domain_allowlist: list[str] | None = None  # for api_call tool


@dataclass
class TaskResult:
    """Final result of a completed task."""

    task_id: str
    success: bool
    summary: str
    steps: list[StepResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    attestation_hash: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class StepEvent:
    """Real-time event emitted during task execution (for SSE streaming)."""

    task_id: str
    step_number: int
    event_type: str  # "thinking", "tool_call", "tool_result", "complete", "error"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
