"""Tests for the XML tool call parser and system prompt builder."""

from __future__ import annotations

import pytest

from enclave.agent.planner import Planner, _parse_single_tool_call
from enclave.agent.models import ToolCall


class TestParseToolCalls:
    """Test parsing <tool_call> blocks from LLM responses."""

    def test_single_tool_call(self) -> None:
        response = """
<thinking>I need to write a hello world file.</thinking>
<tool_call>
  <name>file_ops</name>
  <args>
    <operation>write</operation>
    <path>hello.py</path>
    <content>print("Hello, World!")</content>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "file_ops"
        assert calls[0].args["operation"] == "write"
        assert calls[0].args["path"] == "hello.py"
        assert "Hello" in calls[0].args["content"]

    def test_multiple_tool_calls(self) -> None:
        response = """
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>print(1+1)</code>
  </args>
</tool_call>

Some reasoning text here.

<tool_call>
  <name>file_ops</name>
  <args>
    <operation>read</operation>
    <path>output.txt</path>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].name == "code_exec"
        assert calls[1].name == "file_ops"

    def test_no_tool_calls(self) -> None:
        response = "Just some regular text without any tool calls."
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 0

    def test_empty_response(self) -> None:
        calls = Planner.parse_tool_calls("")
        assert len(calls) == 0

    def test_malformed_xml_missing_name(self) -> None:
        response = """
<tool_call>
  <args>
    <operation>read</operation>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 0

    def test_malformed_xml_no_args(self) -> None:
        response = """
<tool_call>
  <name>file_ops</name>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "file_ops"
        assert calls[0].args == {}

    def test_invalid_tool_name_special_chars(self) -> None:
        response = """
<tool_call>
  <name>../../etc/passwd</name>
  <args></args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 0  # Rejected by snake_case validation

    def test_invalid_tool_name_uppercase(self) -> None:
        response = """
<tool_call>
  <name>FileOps</name>
  <args></args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 0

    def test_boolean_arg_parsing(self) -> None:
        response = """
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>print(1)</code>
    <verbose>true</verbose>
    <debug>false</debug>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].args["verbose"] is True
        assert calls[0].args["debug"] is False

    def test_integer_arg_parsing(self) -> None:
        response = """
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>x = 1</code>
    <timeout>30</timeout>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].args["timeout"] == 30
        assert isinstance(calls[0].args["timeout"], int)

    def test_ampersand_in_content(self) -> None:
        """LLMs often produce unescaped ampersands in XML."""
        response = """
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>x = 1 &amp; 2</code>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1

    def test_injection_attempt_override_system(self) -> None:
        """Attempt to inject system prompt override via tool name."""
        response = """
<tool_call>
  <name>code_exec</name>
  <args>
    <language>python</language>
    <code>import os; os.system("rm -rf /")</code>
  </args>
</tool_call>
"""
        calls = Planner.parse_tool_calls(response)
        assert len(calls) == 1
        # The tool call is parsed but would be sandboxed by the executor
        assert calls[0].name == "code_exec"


class TestParseTaskComplete:
    """Test parsing <task_complete> blocks."""

    def test_task_complete_with_summary(self) -> None:
        response = """
<thinking>All done.</thinking>
<task_complete>
  <summary>Created hello.py with Hello World output.</summary>
</task_complete>
"""
        result = Planner.parse_task_complete(response)
        assert result is not None
        assert "hello.py" in result.summary

    def test_task_complete_without_summary_tag(self) -> None:
        response = """
<task_complete>
Task finished successfully.
</task_complete>
"""
        result = Planner.parse_task_complete(response)
        assert result is not None
        assert "successfully" in result.summary

    def test_no_task_complete(self) -> None:
        response = "Just some regular text."
        result = Planner.parse_task_complete(response)
        assert result is None

    def test_empty_task_complete(self) -> None:
        response = "<task_complete></task_complete>"
        result = Planner.parse_task_complete(response)
        assert result is not None


class TestFormatToolOutputs:
    """Test XML formatting of tool outputs."""

    def test_format_success_output(self) -> None:
        from enclave.agent.models import ToolOutput

        output = ToolOutput(success=True, result="File written successfully")
        xml = Planner.format_tool_outputs_xml("file_ops", output)
        assert "<name>file_ops</name>" in xml
        assert "<success>true</success>" in xml
        assert "File written successfully" in xml

    def test_format_error_output(self) -> None:
        from enclave.agent.models import ToolOutput

        output = ToolOutput(success=False, error="File not found")
        xml = Planner.format_tool_outputs_xml("file_ops", output)
        assert "<success>false</success>" in xml
        assert "<error>File not found</error>" in xml

    def test_format_escapes_special_chars(self) -> None:
        from enclave.agent.models import ToolOutput

        output = ToolOutput(success=True, result="x < 5 && y > 3")
        xml = Planner.format_tool_outputs_xml("code_exec", output)
        assert "&lt;" in xml
        assert "&gt;" in xml
        assert "&amp;" in xml


class TestBuildSystemPrompt:
    """Test system prompt construction."""

    def test_prompt_contains_task(self) -> None:
        from enclave.agent.llm_client import MockLLMClient
        from enclave.tools.base import ToolRegistry

        registry = ToolRegistry()
        planner = Planner(MockLLMClient(), registry)
        prompt = planner.build_system_prompt("Write a hello world program")
        assert "Write a hello world program" in prompt
        assert "Enclave Agent" in prompt

    def test_prompt_contains_tool_schemas(self) -> None:
        from enclave.agent.llm_client import MockLLMClient
        from enclave.tools.base import ToolRegistry
        from enclave.tools.code_executor import CodeExecutor

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        planner = Planner(MockLLMClient(), registry)
        prompt = planner.build_system_prompt("test task")
        assert "code_exec" in prompt
        assert "<tool" in prompt
