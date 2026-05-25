"""Tests for the new coding assistant tools and conversational turn logic."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
import pytest

from enclave.agent.models import ToolCall, Message
from enclave.agent.controller import AgentController
from enclave.agent.llm_client import LLMClient
from enclave.agent.models import LLMResponse, TextBlock, ToolUseBlock
from enclave.tools.base import ToolRegistry
from enclave.tools.read_file import ReadFileTool
from enclave.tools.write_file import WriteFileTool
from enclave.tools.edit_file import EditFileTool
from enclave.tools.list_dir import ListDirTool
from enclave.tools.grep_search import GrepSearchTool
from enclave.tools.bash_tool import BashTool


class MockLLMClient(LLMClient):
    """Mock LLM Client for testing conversation turns."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.call_count = 0
        self.last_messages = []

    async def call(self, **kwargs) -> LLMResponse:
        self.last_messages = kwargs.get("messages", [])
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # Default simple text response
        return LLMResponse(content=[TextBlock(text="Default mock response")])


@pytest.fixture
def temp_project():
    """Create a temporary project structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create some files and directories
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("import os\n\ndef run():\n    print('Hello World')\n", encoding="utf-8")
        (root / "src" / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (root / "README.md").write_text("# Test Project\nThis is a test project.\n", encoding="utf-8")
        # Create a binary file
        with open(root / "binary.bin", "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x00")
        yield root


@pytest.mark.asyncio
async def test_read_file_tool(temp_project) -> None:
    tool = ReadFileTool(project_root=temp_project)
    
    # Read normal text file
    result = await tool.run(path="README.md")
    assert result.success is True
    assert "# Test Project" in result.result
    assert "1 | # Test Project" in result.result

    # Read normal text file with line range
    result = await tool.run(path="src/main.py", start_line=3, end_line=4)
    assert result.success is True
    assert "run()" in result.result
    assert "1 |" not in result.result  # Line 3 is line 3, check line number prefix format

    # Read binary file
    result = await tool.run(path="binary.bin")
    assert result.success is True
    assert "[Binary file:" in result.result

    # Non-existent file
    result = await tool.run(path="missing.txt")
    assert result.success is False
    assert "File not found" in result.error


@pytest.mark.asyncio
async def test_write_file_tool(temp_project) -> None:
    tool = WriteFileTool(project_root=temp_project)

    # Write new file in new nested directory
    result = await tool.run(path="docs/index.md", content="# Index\n")
    assert result.success is True
    assert "docs/index.md" in result.result
    assert (temp_project / "docs" / "index.md").exists()
    assert (temp_project / "docs" / "index.md").read_text(encoding="utf-8") == "# Index\n"


@pytest.mark.asyncio
async def test_edit_file_tool(temp_project) -> None:
    tool = EditFileTool(project_root=temp_project)

    # Valid edit
    result = await tool.run(
        path="src/utils.py",
        old_str="def add(a, b):\n    return a + b\n",
        new_str="def add(a, b):\n    return a + b + 1\n",
    )
    assert result.success is True
    assert "Edited" in result.result
    assert "a/src/utils.py" in result.result
    assert (temp_project / "src" / "utils.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b + 1\n"

    # No match edit
    result = await tool.run(
        path="src/utils.py",
        old_str="non_existent",
        new_str="whatever",
    )
    assert result.success is False
    assert "not found" in result.error

    # Multiple match edit (write another file first)
    (temp_project / "duplicate.txt").write_text("hello\nhello\n", encoding="utf-8")
    result = await tool.run(
        path="duplicate.txt",
        old_str="hello",
        new_str="world",
    )
    assert result.success is False
    assert "matches 2 locations" in result.error


@pytest.mark.asyncio
async def test_list_dir_tool(temp_project) -> None:
    tool = ListDirTool(project_root=temp_project)

    result = await tool.run(path="src")
    assert result.success is True
    assert "main.py" in result.result
    assert "utils.py" in result.result

    result = await tool.run(path=".")
    assert result.success is True
    assert "src/" in result.result
    assert "README.md" in result.result


@pytest.mark.asyncio
async def test_grep_search_tool(temp_project) -> None:
    tool = GrepSearchTool(project_root=temp_project)

    result = await tool.run(pattern="def run")
    assert result.success is True
    assert str(Path("src/main.py")) in result.result
    assert "def run():" in result.result

    result = await tool.run(pattern="non_existent_pattern")
    assert result.success is True
    assert "No matches found" in result.result


@pytest.mark.asyncio
async def test_bash_tool(temp_project) -> None:
    tool = BashTool(cwd=temp_project)

    if sys.platform == "win32":
        cmd = "Write-Output 'Hello from PowerShell'"
    else:
        cmd = "echo 'Hello from PowerShell'"

    result = await tool.run(command=cmd)
    assert result.success is True
    assert "Hello from PowerShell" in result.result


@pytest.mark.asyncio
async def test_controller_run_conversation_turn(temp_project) -> None:
    # Set up mock LLM responses
    # Turn 1 response will invoke a tool call to write a file
    # Turn 2 response will complete the request
    tool_use = ToolUseBlock(
        id="call_1",
        name="write_file",
        input={"path": "new_file.txt", "content": "Hello enclave"},
    )
    r1 = LLMResponse(content=[TextBlock(text="Let me write a file first."), tool_use])
    r2 = LLMResponse(content=[TextBlock(text="All done writing the file.")])

    llm = MockLLMClient(responses=[r1, r2])
    registry = ToolRegistry()
    registry.register(WriteFileTool(project_root=temp_project))

    controller = AgentController(llm, registry)

    history: list[Message] = []
    system_prompt = "You are a coding assistant."

    response_text, updated_history, cost = await controller.run_conversation_turn(
        user_message="Please create a new_file.txt with some greetings.",
        conversation_history=history,
        system_prompt=system_prompt,
    )

    assert "All done" in response_text
    assert len(updated_history) == 4
    # Check messages roles: user -> assistant (tool call) -> user (tool result) -> assistant (final response)
    assert updated_history[0].role == "user"
    assert updated_history[1].role == "assistant"
    assert updated_history[2].role == "user"
    assert updated_history[3].role == "assistant"

    # Verify tool actually executed
    assert (temp_project / "new_file.txt").exists()
    assert (temp_project / "new_file.txt").read_text(encoding="utf-8") == "Hello enclave"
