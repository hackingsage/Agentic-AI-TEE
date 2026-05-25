"""Read file tool — read file contents with optional line ranges.

Supports line-numbered output and binary file detection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_LINES = 2000
MAX_FILE_BYTES = 1024 * 1024  # 1MB


class ReadFileTool(BaseTool):
    """Read file contents from the filesystem.

    Supports optional line ranges and adds line numbers to output.
    Detects binary files and returns a message instead of garbled content.
    """

    name = "read_file"
    description = (
        "Read the contents of a file. Returns the file content with line numbers. "
        "You can optionally specify a line range to read only part of the file. "
        "Always read a file before editing it to understand its current contents. "
        "Paths can be absolute or relative to the project root."
    )

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path — absolute paths used as-is, relative paths resolved from project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self._project_root / p).resolve()

    def _is_binary(self, path: Path) -> bool:
        """Check if a file is binary by reading the first 8KB."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            # Check for null bytes — strong binary indicator
            if b"\x00" in chunk:
                return True
            return False
        except Exception:
            return False

    def validate_args(self, args: dict[str, Any]) -> str | None:
        path = args.get("path")
        if not path:
            return "Missing required argument: 'path'"
        if not isinstance(path, str):
            return "'path' must be a string"

        start = args.get("start_line")
        end = args.get("end_line")
        if start is not None and not isinstance(start, int):
            return "'start_line' must be an integer"
        if end is not None and not isinstance(end, int):
            return "'end_line' must be an integer"
        if start is not None and end is not None and start > end:
            return "'start_line' must be <= 'end_line'"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        path_str: str = kwargs["path"]
        start_line: int | None = kwargs.get("start_line")
        end_line: int | None = kwargs.get("end_line")

        resolved = self._resolve_path(path_str)

        if not resolved.exists():
            return ToolOutput(success=False, error=f"File not found: '{path_str}'")
        if not resolved.is_file():
            return ToolOutput(success=False, error=f"Not a file: '{path_str}' (is it a directory?)")

        # Check file size
        file_size = resolved.stat().st_size
        if file_size > MAX_FILE_BYTES:
            return ToolOutput(
                success=False,
                error=f"File too large ({file_size:,} bytes). Max: {MAX_FILE_BYTES:,} bytes. "
                      f"Use start_line/end_line to read a portion.",
            )

        # Check for binary
        if self._is_binary(resolved):
            return ToolOutput(
                success=True,
                result=f"[Binary file: {resolved.name}, {file_size:,} bytes]",
            )

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to read file: {exc}")

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply line range
        if start_line is not None or end_line is not None:
            s = max(1, start_line or 1) - 1  # Convert to 0-indexed
            e = min(total_lines, end_line or total_lines)
            lines = lines[s:e]
            line_offset = s
        else:
            line_offset = 0
            # Cap total lines
            if len(lines) > MAX_LINES:
                lines = lines[:MAX_LINES]

        # Format with line numbers
        width = len(str(line_offset + len(lines)))
        numbered = []
        for i, line in enumerate(lines):
            line_num = line_offset + i + 1
            numbered.append(f"{line_num:>{width}} | {line}")

        result = "\n".join(numbered)

        # Add metadata
        meta = f"\n\n[File: {path_str} | Total lines: {total_lines}]"
        if start_line is not None or end_line is not None:
            meta += f" [Showing lines {line_offset + 1}-{line_offset + len(lines)}]"
        elif total_lines > MAX_LINES:
            meta += f" [Truncated to first {MAX_LINES} lines — use start_line/end_line for more]"

        return ToolOutput(success=True, result=result + meta)

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (absolute or relative to project root)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, inclusive). Optional.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive). Optional.",
                    },
                },
                "required": ["path"],
            },
        }
