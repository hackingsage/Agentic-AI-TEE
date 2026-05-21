"""Core data structures for the Enclave agent.

All inter-component data is passed as dataclasses — no raw dicts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A parsed tool invocation from the LLM response."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    """Result of a tool execution."""

    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Message:
    """A single message in the conversation history."""

    role: str  # "user", "assistant", or "tool_result"
    content: str


@dataclass
class LLMResponse:
    """Response from the LLM provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        """Estimate cost using Claude Opus pricing ($15/1M input, $75/1M output)."""
        return (self.input_tokens * 15 + self.output_tokens * 75) / 1_000_000


@dataclass
class TaskComplete:
    """Parsed <task_complete> block from the LLM."""

    summary: str


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
        return (self.input_tokens * 15 + self.output_tokens * 75) / 1_000_000


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
