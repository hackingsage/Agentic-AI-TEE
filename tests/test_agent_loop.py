"""Tests for the full agent loop with mock LLM."""

from __future__ import annotations

import pytest

from enclave.agent.controller import AgentController
from enclave.agent.llm_client import MockLLMClient
from enclave.agent.models import TaskRequest
from enclave.tools.base import ToolRegistry
from enclave.tools.code_executor import CodeExecutor
from enclave.tools.file_ops import FileSystem
from enclave.tools.memory_tool import MemoryTool


def _build_controller(
    responses: list[str],
    *,
    max_steps: int = 10,
) -> tuple[AgentController, ToolRegistry]:
    """Helper to build a controller with mock LLM and tools."""
    mock_llm = MockLLMClient(responses=responses)
    registry = ToolRegistry()
    registry.register(CodeExecutor())
    registry.register(FileSystem())
    registry.register(MemoryTool())

    controller = AgentController(mock_llm, registry)
    return controller, registry


class TestAgentLoop:
    """Integration tests for the full agent step loop."""

    @pytest.mark.asyncio
    async def test_simple_task_complete(self) -> None:
        """LLM immediately completes the task."""
        responses = [
            '<thinking>Simple task.</thinking>\n<task_complete>\n  <summary>Done!</summary>\n</task_complete>'
        ]
        controller, _ = _build_controller(responses)

        task = TaskRequest(description="Say hello", max_steps=5)
        result = await controller.run_task(task)

        assert result.success is True
        assert "Done!" in result.summary
        assert len(result.steps) == 1
        assert result.attestation_hash != ""

    @pytest.mark.asyncio
    async def test_tool_call_then_complete(self) -> None:
        """LLM makes a tool call, then completes."""
        responses = [
            # Step 1: tool call
            """<thinking>I'll run some code.</thinking>
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>print("hello from agent")</code>
  </args>
</tool_call>""",
            # Step 2: complete
            """<thinking>Code ran successfully.</thinking>
<task_complete>
  <summary>Executed Python code that prints hello.</summary>
</task_complete>""",
        ]
        controller, _ = _build_controller(responses)

        task = TaskRequest(description="Run hello code", max_steps=10)
        result = await controller.run_task(task)

        assert result.success is True
        assert len(result.steps) >= 2
        # First step should have a tool call
        assert result.steps[0].tool_name == "code_exec"
        assert result.steps[0].output is not None
        assert result.steps[0].output.success is True

    @pytest.mark.asyncio
    async def test_file_write_and_complete(self) -> None:
        """LLM writes a file, then completes."""
        responses = [
            """<thinking>Write a file.</thinking>
<tool_call>
  <name>file_ops</name>
  <args>
    <operation>write</operation>
    <path>hello.py</path>
    <content>print("Hello, World!")</content>
  </args>
</tool_call>""",
            """<task_complete>
  <summary>Created hello.py with a Hello World program.</summary>
</task_complete>""",
        ]
        controller, _ = _build_controller(responses)

        task = TaskRequest(description="Write hello world file")
        result = await controller.run_task(task)

        assert result.success is True
        assert result.steps[0].tool_name == "file_ops"
        assert result.steps[0].output.success is True

    @pytest.mark.asyncio
    async def test_max_steps_guard(self) -> None:
        """Agent is stopped when max steps reached."""
        # Responses that never complete (each makes a tool call to memory_tool to keep loop active)
        responses = [
            """<thinking>Thinking...</thinking>
<tool_call>
  <name>memory_tool</name>
  <args>
    <action>read</action>
    <key>dummy</key>
  </args>
</tool_call>"""
            for _ in range(10)
        ]
        controller, _ = _build_controller(responses, max_steps=3)

        task = TaskRequest(description="Infinite loop task", max_steps=3)
        result = await controller.run_task(task)

        assert result.success is False
        assert "max steps" in result.error.lower() or "max_steps" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cost_guard(self) -> None:
        """Agent is stopped when cost budget exceeded."""
        # Create a mock LLM that returns expensive responses
        mock_llm = MockLLMClient(responses=[
            "Still thinking..." for _ in range(100)
        ])
        # Override call to return high token counts
        original_call = mock_llm.call

        async def expensive_call(*args, **kwargs):
            from enclave.agent.models import LLMResponse, ToolUseBlock
            mock_llm._call_count += 1
            mock_llm._calls.append({"args": args, "kwargs": kwargs})
            return LLMResponse(
                content=[
                    ToolUseBlock(
                        id="call_1",
                        name="code_exec",
                        input={"language": "python", "code": "print(1)"},
                    )
                ],
                input_tokens=100000,  # Very expensive
                output_tokens=50000,
                latency_ms=100.0,
            )

        mock_llm.call = expensive_call

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        controller = AgentController(mock_llm, registry)

        task = TaskRequest(
            description="Expensive task",
            budget_usd=0.01,  # Very low budget
            max_steps=50,
        )
        result = await controller.run_task(task)

        assert result.success is False
        assert "cost" in result.error.lower() or "budget" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_guard(self) -> None:
        """Agent is stopped when timeout reached."""
        import asyncio

        # Create a mock LLM that sleeps
        mock_llm = MockLLMClient()
        original_call = mock_llm.call

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(5)
            return await original_call(*args, **kwargs)

        mock_llm.call = slow_call

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        controller = AgentController(mock_llm, registry)

        task = TaskRequest(
            description="Slow task",
            timeout_seconds=1.0,
            max_steps=50,
        )
        result = await controller.run_task(task)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self) -> None:
        """LLM requests an unknown tool — agent handles gracefully."""
        responses = [
            """<tool_call>
  <name>nonexistent_tool</name>
  <args>
    <data>test</data>
  </args>
</tool_call>""",
            """<task_complete>
  <summary>Completed despite unknown tool.</summary>
</task_complete>""",
        ]
        controller, _ = _build_controller(responses)

        task = TaskRequest(description="Test unknown tool")
        result = await controller.run_task(task)

        # Should still complete (tool error fed back to LLM)
        assert result.success is True
        # The unknown tool step should have failed
        tool_step = result.steps[0]
        assert tool_step.output is not None
        assert tool_step.output.success is False

    @pytest.mark.asyncio
    async def test_event_streaming(self) -> None:
        """Events are emitted during task execution."""
        responses = [
            """<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>print("test")</code>
  </args>
</tool_call>""",
            """<task_complete>
  <summary>Done.</summary>
</task_complete>""",
        ]
        controller, _ = _build_controller(responses)
        event_queue = controller.enable_streaming()

        task = TaskRequest(description="Stream test")
        result = await controller.run_task(task)

        # Collect all events
        events = []
        while not event_queue.empty():
            events.append(await event_queue.get())

        assert len(events) > 0
        event_types = [e.event_type for e in events]
        assert "thinking" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    async def test_attestation_hash_deterministic(self) -> None:
        """Same task + same responses = same attestation hash (deterministic)."""
        responses = [
            '<task_complete><summary>Done</summary></task_complete>'
        ]
        controller1, _ = _build_controller(responses[:])
        controller2, _ = _build_controller(responses[:])

        task1 = TaskRequest(description="Deterministic test", task_id="fixed_id")
        task2 = TaskRequest(description="Deterministic test", task_id="fixed_id")

        result1 = await controller1.run_task(task1)
        result2 = await controller2.run_task(task2)

        assert result1.attestation_hash == result2.attestation_hash
