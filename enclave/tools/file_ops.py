"""File system tool — read/write/list/delete files in an encrypted workspace.

All file operations are scoped to a per-task workspace directory.
Path traversal attacks are prevented by canonicalizing all paths.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_READ_BYTES = 100 * 1024  # 100KB
MAX_WRITE_BYTES = 500 * 1024  # 500KB
VALID_OPERATIONS = {"read", "write", "list", "delete", "exists"}


class FileSystem(BaseTool):
    """File operations scoped to a per-task encrypted workspace.

    Supports read, write, list, delete, and exists operations.
    All paths are validated against traversal attacks.
    """

    name = "file_ops"
    description = (
        "Perform file operations in the task workspace. Operations: read, write, list, "
        "delete, exists. All paths are relative to the workspace root. Cannot access "
        "files outside the workspace."
    )

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="enclave_fs_"))
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def workspace_dir(self) -> Path:
        return self._workspace_dir

    def _resolve_safe_path(self, relative_path: str) -> Path | None:
        """Resolve a relative path safely within the workspace.

        Returns None if the path would escape the workspace (traversal attack).
        """
        # Normalize and remove leading slashes
        clean = relative_path.lstrip("/").lstrip("\\")
        candidate = (self._workspace_dir / clean).resolve()

        # Verify it's still within workspace
        try:
            candidate.relative_to(self._workspace_dir.resolve())
            return candidate
        except ValueError:
            return None

    def validate_args(self, args: dict[str, Any]) -> str | None:
        operation = args.get("operation")
        if not operation:
            return "Missing required argument: 'operation'"
        if operation not in VALID_OPERATIONS:
            return f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
        if operation != "list":
            path = args.get("path")
            if not path:
                return "Missing required argument: 'path'"
            if not isinstance(path, str):
                return "'path' must be a string"
        if operation == "write":
            content = args.get("content")
            if content is None:
                return "Missing required argument: 'content' for write operation"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        operation: str = kwargs["operation"]
        path: str = kwargs.get("path", ".")

        if operation == "read":
            return await self._read(path)
        elif operation == "write":
            return await self._write(path, kwargs["content"])
        elif operation == "list":
            return await self._list(path)
        elif operation == "delete":
            return await self._delete(path)
        elif operation == "exists":
            return await self._exists(path)
        else:
            return ToolOutput(success=False, error=f"Unknown operation: {operation}")

    async def _read(self, path: str) -> ToolOutput:
        safe_path = self._resolve_safe_path(path)
        if safe_path is None:
            return ToolOutput(success=False, error=f"Path traversal denied: '{path}'")

        if not safe_path.exists():
            return ToolOutput(success=False, error=f"File not found: '{path}'")
        if not safe_path.is_file():
            return ToolOutput(success=False, error=f"Not a file: '{path}'")

        try:
            content = safe_path.read_text(encoding="utf-8")
            if len(content) > MAX_READ_BYTES:
                content = content[:MAX_READ_BYTES] + "\n... (truncated)"
            return ToolOutput(success=True, result=content)
        except Exception as exc:
            return ToolOutput(success=False, error=f"Read failed: {exc}")

    async def _write(self, path: str, content: str) -> ToolOutput:
        safe_path = self._resolve_safe_path(path)
        if safe_path is None:
            return ToolOutput(success=False, error=f"Path traversal denied: '{path}'")

        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolOutput(
                success=False,
                error=f"Content exceeds max write size ({MAX_WRITE_BYTES} bytes)",
            )

        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding="utf-8")
            return ToolOutput(
                success=True,
                result=f"Written {len(content)} chars to '{path}'",
            )
        except Exception as exc:
            return ToolOutput(success=False, error=f"Write failed: {exc}")

    async def _list(self, path: str) -> ToolOutput:
        safe_path = self._resolve_safe_path(path)
        if safe_path is None:
            return ToolOutput(success=False, error=f"Path traversal denied: '{path}'")

        if not safe_path.exists():
            return ToolOutput(success=False, error=f"Directory not found: '{path}'")

        target = safe_path if safe_path.is_dir() else self._workspace_dir

        try:
            entries: list[str] = []
            for entry in sorted(target.rglob("*")):
                rel = entry.relative_to(self._workspace_dir)
                prefix = "📁 " if entry.is_dir() else "📄 "
                size_info = f" ({entry.stat().st_size} bytes)" if entry.is_file() else ""
                entries.append(f"{prefix}{rel}{size_info}")

            if not entries:
                return ToolOutput(success=True, result="(empty workspace)")

            return ToolOutput(success=True, result="\n".join(entries[:200]))
        except Exception as exc:
            return ToolOutput(success=False, error=f"List failed: {exc}")

    async def _delete(self, path: str) -> ToolOutput:
        safe_path = self._resolve_safe_path(path)
        if safe_path is None:
            return ToolOutput(success=False, error=f"Path traversal denied: '{path}'")

        if not safe_path.exists():
            return ToolOutput(success=False, error=f"File not found: '{path}'")

        try:
            if safe_path.is_file():
                safe_path.unlink()
            elif safe_path.is_dir():
                import shutil

                shutil.rmtree(safe_path)
            return ToolOutput(success=True, result=f"Deleted '{path}'")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Delete failed: {exc}")

    async def _exists(self, path: str) -> ToolOutput:
        safe_path = self._resolve_safe_path(path)
        if safe_path is None:
            return ToolOutput(success=False, error=f"Path traversal denied: '{path}'")

        exists = safe_path.exists()
        file_type = "directory" if safe_path.is_dir() else "file" if safe_path.is_file() else "none"
        return ToolOutput(success=True, result={"exists": exists, "type": file_type})

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete", "exists"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative file path (not required for 'list' of root)",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (required for 'write' operation)",
                    },
                },
                "required": ["operation"],
            },
        }
