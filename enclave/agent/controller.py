"""Agent Controller — manages the full task lifecycle.

Runs the step loop: LLM call → parse tool calls → dispatch → feed results back.
Enforces max-step, cost, and timeout guards.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, AsyncIterator

from enclave.agent.llm_client import LLMClient
from enclave.agent.models import (
    Message,
    StepEvent,
    StepResult,
    TaskComplete,
    TaskRequest,
    TaskResult,
    ToolCall,
)
from enclave.agent.planner import Planner
from enclave.tools.base import ToolRegistry
from enclave.tools.router import ToolRouter

logger = logging.getLogger(__name__)

# Default guard limits
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_COST_USD = 5.0
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_RETRIES_PER_STEP = 3


class AgentController:
    """Manages the full lifecycle of an agent task.

    Runs the core step loop:
    1. Call LLM via Planner to get next action
    2. Parse tool calls from response
    3. Dispatch tool calls via ToolRouter
    4. Feed results back to LLM
    5. Repeat until task_complete or guard triggers

    Emits StepEvents for real-time streaming.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._planner = Planner(llm_client, tool_registry)
        self._router = ToolRouter(tool_registry)
        self._tool_registry = tool_registry
        self._default_timeout = default_timeout

        # Event subscribers for streaming
        self._event_queue: asyncio.Queue[StepEvent] | None = None

    def enable_streaming(self) -> asyncio.Queue[StepEvent]:
        """Enable event streaming. Returns the queue to read events from."""
        self._event_queue = asyncio.Queue()
        return self._event_queue

    async def _emit_event(self, event: StepEvent) -> None:
        """Emit a step event to the streaming queue."""
        if self._event_queue is not None:
            await self._event_queue.put(event)

    async def run_task(self, task: TaskRequest) -> TaskResult:
        """Execute a task through the full agent loop.

        Args:
            task: The task request with description, budget, and limits.

        Returns:
            TaskResult with all step results and final summary.
        """
        start_time = time.monotonic()
        steps: list[StepResult] = []
        total_cost = 0.0
        messages: list[Message] = [Message(role="user", content=task.description)]
        task_hash_parts: list[str] = [task.description]

        # Build system prompt once
        system_prompt = self._planner.build_system_prompt(task.description)

        max_steps = task.max_steps or DEFAULT_MAX_STEPS
        max_cost = task.budget_usd or DEFAULT_MAX_COST_USD
        timeout = task.timeout_seconds or self._default_timeout

        logger.info(
            "task_started",
            extra={
                "task_id": task.task_id,
                "max_steps": max_steps,
                "budget_usd": max_cost,
                "timeout_s": timeout,
            },
        )

        try:
            result = await asyncio.wait_for(
                self._run_step_loop(
                    task=task,
                    system_prompt=system_prompt,
                    messages=messages,
                    steps=steps,
                    total_cost_ref=[total_cost],
                    max_steps=max_steps,
                    max_cost=max_cost,
                    task_hash_parts=task_hash_parts,
                ),
                timeout=timeout,
            )
            elapsed = time.monotonic() - start_time
            result.elapsed_seconds = elapsed
            return result

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "task_timeout",
                extra={
                    "task_id": task.task_id,
                    "elapsed_s": round(elapsed, 2),
                    "steps_completed": len(steps),
                },
            )
            await self._emit_event(StepEvent(
                task_id=task.task_id,
                step_number=len(steps),
                event_type="error",
                data={"error": f"Task timed out after {timeout}s"},
            ))
            return TaskResult(
                task_id=task.task_id,
                success=False,
                summary=f"Task timed out after {timeout}s ({len(steps)} steps completed)",
                steps=steps,
                total_cost_usd=sum(s.cost_usd for s in steps),
                error="timeout",
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "task_error",
                extra={
                    "task_id": task.task_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            await self._emit_event(StepEvent(
                task_id=task.task_id,
                step_number=len(steps),
                event_type="error",
                data={"error": str(exc)},
            ))
            return TaskResult(
                task_id=task.task_id,
                success=False,
                summary=f"Task failed: {exc}",
                steps=steps,
                total_cost_usd=sum(s.cost_usd for s in steps),
                error=str(exc),
                elapsed_seconds=elapsed,
            )

    async def _run_step_loop(
        self,
        *,
        task: TaskRequest,
        system_prompt: str,
        messages: list[Message],
        steps: list[StepResult],
        total_cost_ref: list[float],
        max_steps: int,
        max_cost: float,
        task_hash_parts: list[str],
    ) -> TaskResult:
        """Inner step loop — separated for timeout wrapping."""

        for step_num in range(1, max_steps + 1):
            # --- Guard: cost ---
            current_cost = sum(s.cost_usd for s in steps)
            if current_cost >= max_cost:
                logger.warning(
                    "cost_guard_triggered",
                    extra={
                        "task_id": task.task_id,
                        "cost_usd": round(current_cost, 4),
                        "budget_usd": max_cost,
                    },
                )
                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="error",
                    data={"error": f"Cost budget exceeded: ${current_cost:.4f} >= ${max_cost}"},
                ))
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    summary=f"Cost budget exceeded (${current_cost:.4f} >= ${max_cost})",
                    steps=steps,
                    total_cost_usd=current_cost,
                    error="cost_exceeded",
                )

            # --- Step 1: Call LLM ---
            await self._emit_event(StepEvent(
                task_id=task.task_id,
                step_number=step_num,
                event_type="thinking",
                data={"status": "calling LLM"},
            ))

            llm_response = await self._planner.plan_next_step(
                system_prompt=system_prompt,
                messages=messages,
                task_id=task.task_id,
                step_number=step_num,
            )

            # --- Step 2: Check for task_complete ---
            task_complete = Planner.parse_task_complete(llm_response.text)
            if task_complete:
                step = StepResult(
                    step_number=step_num,
                    llm_response=llm_response.text,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    latency_ms=llm_response.latency_ms,
                )
                steps.append(step)
                task_hash_parts.append(llm_response.text)

                # Build attestation hash
                attestation_hash = hashlib.sha3_256(
                    "\n".join(task_hash_parts).encode()
                ).hexdigest()

                total_cost = sum(s.cost_usd for s in steps)

                logger.info(
                    "task_completed",
                    extra={
                        "task_id": task.task_id,
                        "steps": len(steps),
                        "cost_usd": round(total_cost, 4),
                        "attestation_hash": attestation_hash[:16],
                    },
                )

                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="complete",
                    data={"summary": task_complete.summary},
                ))

                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    summary=task_complete.summary,
                    steps=steps,
                    total_cost_usd=total_cost,
                    attestation_hash=attestation_hash,
                )

            # --- Step 3: Parse tool calls ---
            tool_calls = Planner.parse_tool_calls(llm_response.text)

            if not tool_calls:
                # No tool call and no task_complete — add response and continue
                messages.append(Message(role="assistant", content=llm_response.text))
                messages.append(Message(
                    role="user",
                    content=(
                        "You did not make a tool call or complete the task. "
                        "Please either use a tool or output <task_complete> with a summary."
                    ),
                ))
                step = StepResult(
                    step_number=step_num,
                    llm_response=llm_response.text,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    latency_ms=llm_response.latency_ms,
                )
                steps.append(step)
                continue

            # --- Step 4: Execute tool calls ---
            for tool_call in tool_calls:
                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="tool_call",
                    data={"tool": tool_call.name, "args_keys": list(tool_call.args.keys())},
                ))

                tool_output = await self._router.dispatch(
                    tool_call,
                    task_id=task.task_id,
                    step_number=step_num,
                )

                # Build step result
                step = StepResult(
                    step_number=step_num,
                    tool_name=tool_call.name,
                    tool_args=tool_call.args,
                    output=tool_output,
                    llm_response=llm_response.text,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    latency_ms=llm_response.latency_ms,
                )
                steps.append(step)
                task_hash_parts.append(f"{tool_call.name}:{tool_output.result}")

                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="tool_result",
                    data={
                        "tool": tool_call.name,
                        "success": tool_output.success,
                        "latency_ms": round(tool_output.latency_ms, 2),
                    },
                ))

                # Feed result back to conversation
                messages.append(Message(role="assistant", content=llm_response.text))
                tool_result_xml = Planner.format_tool_outputs_xml(
                    tool_call.name, tool_output
                )
                messages.append(Message(role="user", content=tool_result_xml))

        # --- Guard: max steps ---
        total_cost = sum(s.cost_usd for s in steps)
        logger.warning(
            "max_steps_guard_triggered",
            extra={
                "task_id": task.task_id,
                "max_steps": max_steps,
                "cost_usd": round(total_cost, 4),
            },
        )
        await self._emit_event(StepEvent(
            task_id=task.task_id,
            step_number=max_steps,
            event_type="error",
            data={"error": f"Max steps ({max_steps}) reached"},
        ))
        return TaskResult(
            task_id=task.task_id,
            success=False,
            summary=f"Max steps ({max_steps}) reached without completion",
            steps=steps,
            total_cost_usd=total_cost,
            error="max_steps_exceeded",
        )
