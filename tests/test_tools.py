"""Tests for individual tool implementations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from enclave.agent.models import ToolOutput


class TestCodeExecutor:
    """Test the code execution tool."""

    @pytest.fixture
    def executor(self):
        from enclave.tools.code_executor import CodeExecutor

        workspace = Path(tempfile.mkdtemp(prefix="test_code_"))
        return CodeExecutor(workspace_dir=workspace)

    @pytest.mark.asyncio
    async def test_python_hello_world(self, executor) -> None:
        result = await executor.run(language="python", code='print("Hello, World!")')
        assert result.success is True
        assert "Hello, World!" in result.result

    @pytest.mark.asyncio
    async def test_python_error(self, executor) -> None:
        result = await executor.run(language="python", code="raise ValueError('test error')")
        assert result.success is False
        assert result.error is not None
        assert "ValueError" in result.error

    @pytest.mark.asyncio
    async def test_bash_execution(self, executor) -> None:
        result = await executor.run(language="bash", code='echo "hello from bash"')
        assert result.success is True
        assert "hello from bash" in result.result

    @pytest.mark.asyncio
    async def test_timeout(self, executor) -> None:
        result = await executor.run(
            language="python",
            code="import time; time.sleep(10)",
            timeout=1,
        )
        assert result.success is False
        assert "timed out" in result.error

    def test_validate_invalid_language(self, executor) -> None:
        error = executor.validate_args({"language": "rust", "code": "fn main() {}"})
        assert error is not None
        assert "Unsupported" in error

    def test_validate_missing_code(self, executor) -> None:
        error = executor.validate_args({"language": "python"})
        assert error is not None
        assert "code" in error.lower()

    def test_validate_valid_args(self, executor) -> None:
        error = executor.validate_args({"language": "python", "code": "x = 1"})
        assert error is None


class TestFileSystem:
    """Test the file system tool."""

    @pytest.fixture
    def fs(self):
        from enclave.tools.file_ops import FileSystem

        workspace = Path(tempfile.mkdtemp(prefix="test_fs_"))
        return FileSystem(workspace_dir=workspace)

    @pytest.mark.asyncio
    async def test_write_and_read(self, fs) -> None:
        write_result = await fs.run(operation="write", path="test.txt", content="hello")
        assert write_result.success is True

        read_result = await fs.run(operation="read", path="test.txt")
        assert read_result.success is True
        assert read_result.result == "hello"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, fs) -> None:
        result = await fs.run(operation="read", path="nonexistent.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_denied(self, fs) -> None:
        result = await fs.run(operation="read", path="../../../etc/passwd")
        assert result.success is False
        assert "traversal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_dotdot(self, fs) -> None:
        result = await fs.run(operation="write", path="../../evil.txt", content="pwned")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_workspace(self, fs) -> None:
        await fs.run(operation="write", path="a.txt", content="a")
        await fs.run(operation="write", path="b.txt", content="b")
        result = await fs.run(operation="list", path=".")
        assert result.success is True
        assert "a.txt" in result.result
        assert "b.txt" in result.result

    @pytest.mark.asyncio
    async def test_delete_file(self, fs) -> None:
        await fs.run(operation="write", path="delete_me.txt", content="temp")
        delete_result = await fs.run(operation="delete", path="delete_me.txt")
        assert delete_result.success is True

        read_result = await fs.run(operation="read", path="delete_me.txt")
        assert read_result.success is False

    @pytest.mark.asyncio
    async def test_exists(self, fs) -> None:
        result = await fs.run(operation="exists", path="nope.txt")
        assert result.success is True
        assert result.result["exists"] is False

        await fs.run(operation="write", path="yes.txt", content="exists")
        result = await fs.run(operation="exists", path="yes.txt")
        assert result.success is True
        assert result.result["exists"] is True

    @pytest.mark.asyncio
    async def test_nested_directory_write(self, fs) -> None:
        result = await fs.run(
            operation="write", path="subdir/deep/file.txt", content="nested"
        )
        assert result.success is True

        read_result = await fs.run(operation="read", path="subdir/deep/file.txt")
        assert read_result.success is True
        assert read_result.result == "nested"

    def test_validate_missing_operation(self, fs) -> None:
        error = fs.validate_args({"path": "test.txt"})
        assert error is not None

    def test_validate_invalid_operation(self, fs) -> None:
        error = fs.validate_args({"operation": "chmod", "path": "test.txt"})
        assert error is not None


class TestAPICallTool:
    """Test the API call tool."""

    @pytest.fixture
    def api_tool(self):
        from enclave.tools.api_call import APICallTool

        return APICallTool(domain_allowlist=["example.com", "api.github.com"])

    @pytest.fixture
    def api_tool_no_allowlist(self):
        from enclave.tools.api_call import APICallTool

        return APICallTool(domain_allowlist=None)

    def test_domain_allowed(self, api_tool) -> None:
        assert api_tool._is_domain_allowed("https://example.com/api") is True
        assert api_tool._is_domain_allowed("https://sub.example.com/api") is True
        assert api_tool._is_domain_allowed("https://api.github.com/repos") is True

    def test_domain_denied(self, api_tool) -> None:
        assert api_tool._is_domain_allowed("https://evil.com/steal") is False
        assert api_tool._is_domain_allowed("https://notexample.com") is False

    def test_no_allowlist_allows_all(self, api_tool_no_allowlist) -> None:
        assert api_tool_no_allowlist._is_domain_allowed("https://anything.com") is True

    def test_validate_invalid_method(self, api_tool) -> None:
        error = api_tool.validate_args({"method": "HACK", "url": "https://example.com"})
        assert error is not None

    def test_validate_invalid_scheme(self, api_tool) -> None:
        error = api_tool.validate_args({"method": "GET", "url": "ftp://example.com"})
        assert error is not None

    def test_validate_valid_args(self, api_tool) -> None:
        error = api_tool.validate_args({"method": "GET", "url": "https://example.com/api"})
        assert error is None

    @pytest.mark.asyncio
    async def test_domain_blocked_at_runtime(self, api_tool) -> None:
        result = await api_tool.run(method="GET", url="https://evil.com/steal")
        assert result.success is False
        assert "not in the allowlist" in result.error


class TestMemoryTool:
    """Test the memory search tool."""

    @pytest.fixture
    def memory_tool(self):
        from enclave.tools.memory_tool import MemoryTool

        return MemoryTool()

    @pytest.mark.asyncio
    async def test_store_and_search(self, memory_tool) -> None:
        await memory_tool.run(
            operation="store",
            content="Python is a programming language created by Guido van Rossum",
        )
        await memory_tool.run(
            operation="store",
            content="Rust is a systems programming language focused on safety",
        )

        result = await memory_tool.run(
            operation="search",
            query="programming language Python",
        )
        assert result.success is True
        assert "Python" in result.result

    @pytest.mark.asyncio
    async def test_list_recent(self, memory_tool) -> None:
        await memory_tool.run(operation="store", content="Entry one")
        await memory_tool.run(operation="store", content="Entry two")

        result = await memory_tool.run(operation="list_recent", n=10)
        assert result.success is True
        assert "Entry one" in result.result
        assert "Entry two" in result.result

    @pytest.mark.asyncio
    async def test_search_empty_store(self, memory_tool) -> None:
        result = await memory_tool.run(operation="search", query="anything")
        assert result.success is True
        assert "No matching" in result.result

    def test_validate_missing_query(self, memory_tool) -> None:
        error = memory_tool.validate_args({"operation": "search"})
        assert error is not None

    def test_validate_missing_content(self, memory_tool) -> None:
        error = memory_tool.validate_args({"operation": "store"})
        assert error is not None


class TestBrowserTool:
    """Test the browser tool with mock backend."""

    @pytest.fixture
    def browser(self):
        from enclave.tools.browser_tool import BrowserTool

        return BrowserTool()

    @pytest.mark.asyncio
    async def test_navigate(self, browser) -> None:
        result = await browser.run(operation="navigate", url="https://example.com")
        assert result.success is True
        assert "example.com" in result.result

    @pytest.mark.asyncio
    async def test_screenshot(self, browser) -> None:
        await browser.run(operation="navigate", url="https://example.com")
        result = await browser.run(operation="screenshot")
        assert result.success is True
        assert result.result.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_get_text(self, browser) -> None:
        await browser.run(operation="navigate", url="https://example.com")
        result = await browser.run(operation="get_text")
        assert result.success is True

    def test_validate_navigate_needs_url(self, browser) -> None:
        error = browser.validate_args({"operation": "navigate"})
        assert error is not None

    def test_validate_click_needs_selector(self, browser) -> None:
        error = browser.validate_args({"operation": "click"})
        assert error is not None


class TestToolRegistry:
    """Test the tool registry."""

    def test_register_and_get(self) -> None:
        from enclave.tools.base import ToolRegistry
        from enclave.tools.code_executor import CodeExecutor

        registry = ToolRegistry()
        executor = CodeExecutor()
        registry.register(executor)

        assert registry.get("code_exec") is executor
        assert registry.get("nonexistent") is None

    def test_duplicate_name_raises(self) -> None:
        from enclave.tools.base import ToolRegistry
        from enclave.tools.code_executor import CodeExecutor

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        with pytest.raises(ValueError, match="Duplicate"):
            registry.register(CodeExecutor())

    def test_invalid_name_raises(self) -> None:
        from enclave.tools.base import BaseTool, ToolRegistry
        from enclave.agent.models import ToolOutput

        class BadTool(BaseTool):
            name = "BadName"
            description = "test"

            async def run(self, **kwargs):
                return ToolOutput(success=True)

            def schema_xml(self):
                return "<tool />"

        registry = ToolRegistry()
        with pytest.raises(ValueError, match="snake_case"):
            registry.register(BadTool())

    def test_build_schema_xml(self) -> None:
        from enclave.tools.base import ToolRegistry
        from enclave.tools.code_executor import CodeExecutor
        from enclave.tools.file_ops import FileSystem

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        registry.register(FileSystem())
        xml = registry.build_schema_xml()
        assert "code_exec" in xml
        assert "file_ops" in xml


class TestToolRouter:
    """Test tool dispatch and validation."""

    @pytest.fixture
    def router(self):
        from enclave.tools.base import ToolRegistry
        from enclave.tools.code_executor import CodeExecutor
        from enclave.tools.router import ToolRouter

        registry = ToolRegistry()
        registry.register(CodeExecutor())
        return ToolRouter(registry)

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, router) -> None:
        from enclave.agent.models import ToolCall

        call = ToolCall(name="nonexistent_tool", args={})
        result = await router.dispatch(call)
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_invalid_args(self, router) -> None:
        from enclave.agent.models import ToolCall

        call = ToolCall(name="code_exec", args={"language": "cobol", "code": "x"})
        result = await router.dispatch(call)
        assert result.success is False
        assert "Invalid arguments" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_valid_call(self, router) -> None:
        from enclave.agent.models import ToolCall

        call = ToolCall(name="code_exec", args={"language": "python", "code": "print(42)"})
        result = await router.dispatch(call, task_id="test", step_number=1)
        assert result.success is True
        assert "42" in result.result
