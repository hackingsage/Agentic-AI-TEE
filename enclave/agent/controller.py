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
    TaskRequest,
    TaskResult,
    ToolCall,
    ToolResultBlock,
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

            async def on_chunk(token: str) -> None:
                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="chunk",
                    data={"chunk": token},
                ))

            llm_response = await self._planner.plan_next_step(
                system_prompt=system_prompt,
                messages=messages,
                task_id=task.task_id,
                step_number=step_num,
                on_chunk=on_chunk,
            )

            # --- Step 2: Check if task is complete (no tool_use blocks = done) ---
            if not llm_response.has_tool_use:
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
                    data={"summary": llm_response.text},
                ))

                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    summary=llm_response.text,
                    steps=steps,
                    total_cost_usd=total_cost,
                    attestation_hash=attestation_hash,
                )

            # --- Step 3: Execute tool calls ---
            # Append assistant message with all blocks (text + tool uses)
            messages.append(Message(role="assistant", content=llm_response.content))

            tool_results: list[ToolResultBlock] = []
            for i, tool_block in enumerate(llm_response.tool_use_blocks):
                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="tool_call",
                    data={"tool": tool_block.name, "args_keys": list(tool_block.input.keys())},
                ))

                tool_call = ToolCall(name=tool_block.name, args=tool_block.input)
                tool_output = await self._router.dispatch(
                    tool_call,
                    task_id=task.task_id,
                    step_number=step_num,
                )

                # Collect result block
                tool_results.append(ToolResultBlock(
                    tool_use_id=tool_block.id,
                    content=str(tool_output.result) if tool_output.success 
                            else f"Error: {tool_output.error}",
                    is_error=not tool_output.success,
                ))

                # Build step result.
                # Only attribute LLM tokens/latency to the first tool result block to avoid duplicate costs.
                step = StepResult(
                    step_number=step_num,
                    tool_name=tool_block.name,
                    tool_args=tool_block.input,
                    output=tool_output,
                    llm_response=llm_response.text if i == 0 else "",
                    input_tokens=llm_response.input_tokens if i == 0 else 0,
                    output_tokens=llm_response.output_tokens if i == 0 else 0,
                    latency_ms=llm_response.latency_ms if i == 0 else 0.0,
                )
                steps.append(step)
                task_hash_parts.append(f"{tool_block.name}:{tool_output.result}")

                await self._emit_event(StepEvent(
                    task_id=task.task_id,
                    step_number=step_num,
                    event_type="tool_result",
                    data={
                        "tool": tool_block.name,
                        "success": tool_output.success,
                        "latency_ms": round(tool_output.latency_ms, 2),
                    },
                ))

            # Feed all tool results back as a single user message containing tool result blocks
            messages.append(Message(role="user", content=tool_results))

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

    async def run_conversation_turn(
        self,
        user_message: str,
        conversation_history: list[Message],
        system_prompt: str,
        *,
        max_steps: int = 25,
        max_cost: float = 2.0,
        task_id: str = "",
        timeout_seconds: float = 300.0,
    ) -> tuple[str, list[Message], float]:
        """Execute one conversation turn (user message → agent response).

        Returns:
            (response_text, updated_history, cost_usd)
        """
        # Append the new user message to conversation history
        conversation_history.append(Message(role="user", content=user_message))

        steps: list[StepResult] = []

        async def _inner_loop():
            for step_num in range(1, max_steps + 1):
                # --- Guard: cost ---
                current_cost = sum(s.cost_usd for s in steps)
                if current_cost >= max_cost:
                    logger.warning(
                        "cost_guard_triggered",
                        extra={
                            "task_id": task_id,
                            "cost_usd": round(current_cost, 4),
                            "budget_usd": max_cost,
                        },
                    )
                    err_msg = f"Cost budget exceeded: ${current_cost:.4f} >= ${max_cost}"
                    await self._emit_event(StepEvent(
                        task_id=task_id,
                        step_number=step_num,
                        event_type="error",
                        data={"error": err_msg},
                    ))
                    return f"Error: {err_msg}", current_cost

                # --- Step 1: Call LLM ---
                await self._emit_event(StepEvent(
                    task_id=task_id,
                    step_number=step_num,
                    event_type="thinking",
                    data={"status": "calling LLM"},
                ))

                async def on_chunk(token: str) -> None:
                    await self._emit_event(StepEvent(
                        task_id=task_id,
                        step_number=step_num,
                        event_type="chunk",
                        data={"chunk": token},
                    ))

                llm_response = await self._planner.plan_next_step(
                    system_prompt=system_prompt,
                    messages=conversation_history,
                    task_id=task_id,
                    step_number=step_num,
                    on_chunk=on_chunk,
                )

                # --- Step 2: Check if task is complete (no tool_use blocks = done) ---
                if not llm_response.has_tool_use:
                    step = StepResult(
                        step_number=step_num,
                        llm_response=llm_response.text,
                        input_tokens=llm_response.input_tokens,
                        output_tokens=llm_response.output_tokens,
                        latency_ms=llm_response.latency_ms,
                    )
                    steps.append(step)

                    # Append assistant's final response message to history
                    conversation_history.append(Message(role="assistant", content=llm_response.content))

                    total_cost = sum(s.cost_usd for s in steps)

                    await self._emit_event(StepEvent(
                        task_id=task_id,
                        step_number=step_num,
                        event_type="complete",
                        data={"summary": llm_response.text},
                    ))

                    return llm_response.text, total_cost

                # --- Step 3: Execute tool calls ---
                # Append assistant message with tool calls to history
                conversation_history.append(Message(role="assistant", content=llm_response.content))

                tool_results: list[ToolResultBlock] = []
                for i, tool_block in enumerate(llm_response.tool_use_blocks):
                    await self._emit_event(StepEvent(
                        task_id=task_id,
                        step_number=step_num,
                        event_type="tool_call",
                        data={"tool": tool_block.name, "args_keys": list(tool_block.input.keys())},
                    ))

                    tool_call = ToolCall(name=tool_block.name, args=tool_block.input)
                    tool_output = await self._router.dispatch(
                        tool_call,
                        task_id=task_id,
                        step_number=step_num,
                    )

                    # Collect result block
                    tool_results.append(ToolResultBlock(
                        tool_use_id=tool_block.id,
                        content=str(tool_output.result) if tool_output.success
                                else f"Error: {tool_output.error}",
                        is_error=not tool_output.success,
                    ))

                    # Build step result
                    # Only attribute LLM tokens/latency to the first tool result block
                    step = StepResult(
                        step_number=step_num,
                        tool_name=tool_block.name,
                        tool_args=tool_block.input,
                        output=tool_output,
                        llm_response=llm_response.text if i == 0 else "",
                        input_tokens=llm_response.input_tokens if i == 0 else 0,
                        output_tokens=llm_response.output_tokens if i == 0 else 0,
                        latency_ms=llm_response.latency_ms if i == 0 else 0.0,
                    )
                    steps.append(step)

                    await self._emit_event(StepEvent(
                        task_id=task_id,
                        step_number=step_num,
                        event_type="tool_result",
                        data={
                            "tool": tool_block.name,
                            "success": tool_output.success,
                            "latency_ms": round(tool_output.latency_ms, 2),
                        },
                    ))

                # Feed all tool results back as a single user message
                conversation_history.append(Message(role="user", content=tool_results))

            # --- Guard: max steps reached ---
            total_cost = sum(s.cost_usd for s in steps)
            err_msg = f"Max steps ({max_steps}) reached without completion"
            await self._emit_event(StepEvent(
                task_id=task_id,
                step_number=max_steps,
                event_type="error",
                data={"error": err_msg},
            ))
            return f"Error: {err_msg}", total_cost

        try:
            resp_text, cost = await asyncio.wait_for(_inner_loop(), timeout=timeout_seconds)
            return resp_text, conversation_history, cost
        except asyncio.TimeoutError:
            err_msg = f"Conversation turn timed out after {timeout_seconds}s"
            await self._emit_event(StepEvent(
                task_id=task_id,
                step_number=len(steps) + 1,
                event_type="error",
                data={"error": err_msg},
            ))
            total_cost = sum(s.cost_usd for s in steps)
            return f"Error: {err_msg}", conversation_history, total_cost
        except Exception as exc:
            logger.error(
                "conversation_turn_error",
                extra={
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            err_msg = f"Conversation turn failed: {exc}"
            await self._emit_event(StepEvent(
                task_id=task_id,
                step_number=len(steps) + 1,
                event_type="error",
                data={"error": err_msg},
            ))
            total_cost = sum(s.cost_usd for s in steps)
            return f"Error: {err_msg}", conversation_history, total_cost

