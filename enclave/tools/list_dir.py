"""List directory tool — list files and subdirectories.

Respects common ignore patterns (.git, node_modules, __pycache__, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Directories to skip when listing
IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    ".tox", ".nox",
    "dist", "build", "*.egg-info",
    ".next", ".nuxt",
    "target",  # Rust/Java
    ".idea", ".vscode",
}

MAX_ENTRIES = 200


class ListDirTool(BaseTool):
    """List directory contents.

    Shows files and subdirectories with type indicators and sizes.
    Skips common ignored directories (.git, node_modules, etc.).
    """

    name = "list_dir"
    description = (
        "List the contents of a directory. Shows files and subdirectories "
        "with their sizes. Skips common ignored directories like .git, "
        "node_modules, __pycache__, .venv, etc. "
        "Defaults to listing the project root directory."
    )

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path — absolute paths used as-is, relative paths resolved from project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self._project_root / p).resolve()

    def _should_ignore(self, name: str) -> bool:
        """Check if a directory name should be ignored."""
        if name in IGNORE_DIRS:
            return True
        # Check glob patterns (e.g. *.egg-info)
        for pattern in IGNORE_DIRS:
            if "*" in pattern:
                import fnmatch
                if fnmatch.fnmatch(name, pattern):
                    return True
        return False

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f}MB"

    def validate_args(self, args: dict[str, Any]) -> str | None:
        # Path is optional, defaults to project root
        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return "'path' must be a string"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        path_str: str = kwargs.get("path", ".")
        resolved = self._resolve_path(path_str)

        if not resolved.exists():
            return ToolOutput(success=False, error=f"Directory not found: '{path_str}'")
        if not resolved.is_dir():
            return ToolOutput(success=False, error=f"Not a directory: '{path_str}'")

        try:
            entries: list[str] = []
            dirs: list[tuple[str, int]] = []
            files: list[tuple[str, int]] = []

            for entry in sorted(resolved.iterdir()):
                name = entry.name

                if entry.is_dir():
                    if self._should_ignore(name):
                        continue
                    # Count immediate children
                    try:
                        child_count = sum(1 for _ in entry.iterdir())
                    except PermissionError:
                        child_count = -1
                    dirs.append((name, child_count))
                elif entry.is_file():
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    files.append((name, size))

            # Format directories first, then files
            for name, child_count in dirs:
                count_str = f" ({child_count} items)" if child_count >= 0 else ""
                entries.append(f"📁 {name}/{count_str}")

            for name, size in files:
                entries.append(f"📄 {name} ({self._format_size(size)})")

            if not entries:
                return ToolOutput(success=True, result=f"(empty directory: {path_str})")

            if len(entries) > MAX_ENTRIES:
                result = "\n".join(entries[:MAX_ENTRIES])
                result += f"\n\n... and {len(entries) - MAX_ENTRIES} more entries"
            else:
                result = "\n".join(entries)

            result += f"\n\n[{len(dirs)} directories, {len(files)} files in '{path_str}']"

            return ToolOutput(success=True, result=result)

        except PermissionError:
            return ToolOutput(success=False, error=f"Permission denied: '{path_str}'")
        except Exception as exc:
            return ToolOutput(success=False, error=f"List failed: {exc}")

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: project root). Absolute or relative.",
                    },
                },
                "required": [],
            },
        }
