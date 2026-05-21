"""Tool router: dispatches tool calls to the correct handler.

Validates tool names and arguments before execution.
Wraps execution with timeout and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from enclave.agent.models import ToolCall, ToolOutput
from enclave.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)

# Default timeout for any single tool execution
DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0


class ToolRouter:
    """Dispatches ToolCall objects to the correct BaseTool handler.

    - Rejects unknown tool names
    - Validates arguments against tool schema
    - Wraps execution with timeout and error handling
    - Logs structured execution metadata (never the actual data)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._registry = registry
        self._default_timeout = default_timeout

    async def dispatch(
        self,
        call: ToolCall,
        *,
        task_id: str = "",
        step_number: int = 0,
        timeout: float | None = None,
    ) -> ToolOutput:
        """Dispatch a tool call to the appropriate handler.

        Args:
            call: The parsed tool call from the LLM.
            task_id: Current task ID for logging.
            step_number: Current step number for logging.
            timeout: Override timeout for this call. Uses default if None.

        Returns:
            ToolOutput with execution result.
        """
        effective_timeout = timeout or self._default_timeout

        # Reject unknown tool names
        tool = self._registry.get(call.name)
        if tool is None:
            logger.warning(
                "unknown_tool_rejected",
                extra={
                    "tool": call.name,
                    "task_id": task_id,
                    "step": step_number,
                    "available_tools": self._registry.names,
                },
            )
            return ToolOutput(
                success=False,
                error=f"Unknown tool '{call.name}'. Available tools: {self._registry.names}",
            )

        # Validate arguments
        validation_error = tool.validate_args(call.args)
        if validation_error:
            logger.warning(
                "tool_args_invalid",
                extra={
                    "tool": call.name,
                    "task_id": task_id,
                    "step": step_number,
                    "error": validation_error,
                },
            )
            return ToolOutput(success=False, error=f"Invalid arguments: {validation_error}")

        # Execute with timeout and error handling
        start_time = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.run(**call.args),
                timeout=effective_timeout,
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            result.latency_ms = elapsed_ms

            logger.info(
                "tool_executed",
                extra={
                    "tool": call.name,
                    "task_id": task_id,
                    "step": step_number,
                    "success": result.success,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "tool_timeout",
                extra={
                    "tool": call.name,
                    "task_id": task_id,
                    "step": step_number,
                    "timeout_seconds": effective_timeout,
                },
            )
            return ToolOutput(
                success=False,
                error=f"Tool '{call.name}' timed out after {effective_timeout}s",
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "tool_error",
                extra={
                    "tool": call.name,
                    "task_id": task_id,
                    "step": step_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return ToolOutput(
                success=False,
                error=f"Tool '{call.name}' failed: {type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms,
            )
