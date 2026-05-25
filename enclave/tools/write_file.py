"""Write file tool — create or overwrite files.

Creates parent directories automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_WRITE_BYTES = 1024 * 1024  # 1MB


class WriteFileTool(BaseTool):
    """Create or fully overwrite a file.

    Creates parent directories automatically.
    Use this for creating new files or completely replacing file contents.
    For partial edits, use the edit_file tool instead.
    """

    name = "write_file"
    description = (
        "Create a new file or overwrite an existing file with the given content. "
        "Parent directories are created automatically if they don't exist. "
        "Use this for creating new files. For editing existing files, prefer the "
        "edit_file tool which does targeted search-and-replace."
    )

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path — absolute paths used as-is, relative paths resolved from project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self._project_root / p).resolve()

    def validate_args(self, args: dict[str, Any]) -> str | None:
        path = args.get("path")
        if not path:
            return "Missing required argument: 'path'"
        if not isinstance(path, str):
            return "'path' must be a string"
        content = args.get("content")
        if content is None:
            return "Missing required argument: 'content'"
        if not isinstance(content, str):
            return "'content' must be a string"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        path_str: str = kwargs["path"]
        content: str = kwargs["content"]

        # Size check
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > MAX_WRITE_BYTES:
            return ToolOutput(
                success=False,
                error=f"Content too large ({len(content_bytes):,} bytes). Max: {MAX_WRITE_BYTES:,} bytes.",
            )

        resolved = self._resolve_path(path_str)
        is_new = not resolved.exists()

        try:
            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            resolved.write_text(content, encoding="utf-8")

            action = "Created" if is_new else "Overwritten"
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            return ToolOutput(
                success=True,
                result=f"{action} file '{path_str}' ({len(content_bytes):,} bytes, {line_count} lines)",
            )
        except Exception as exc:
            logger.error(
                "write_file_error",
                extra={"path": path_str, "error": str(exc)},
            )
            return ToolOutput(success=False, error=f"Write failed: {exc}")

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
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        }
