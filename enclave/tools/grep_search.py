"""Grep search tool — search for patterns in code files.

Supports regex patterns, file extension filtering, and context display.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_RESULTS = 50
MAX_LINE_LENGTH = 500

# Directories to always skip
SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env",
    ".tox", ".nox",
    "dist", "build",
    ".next", ".nuxt",
    "target",
    ".idea", ".vscode",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".db", ".sqlite", ".sqlite3",
    ".bin", ".dat",
}


class GrepSearchTool(BaseTool):
    """Search for patterns in code files using regex.

    Supports file extension filtering and skips binary files
    and common ignored directories.
    """

    name = "grep_search"
    description = (
        "Search for a pattern (regex) across files in a directory. "
        "Returns matching lines with file paths and line numbers. "
        "Use the 'include' parameter to filter by file type (e.g., '*.py'). "
        "Skips binary files and common ignored directories (.git, node_modules, etc.). "
        "Results are capped at 50 matches."
    )

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path — absolute paths used as-is, relative paths resolved from project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (self._project_root / p).resolve()

    def _should_skip_dir(self, name: str) -> bool:
        """Check if a directory should be skipped."""
        return name in SKIP_DIRS

    def _should_skip_file(self, path: Path) -> bool:
        """Check if a file should be skipped (binary, etc.)."""
        return path.suffix.lower() in BINARY_EXTENSIONS

    def _matches_include(self, path: Path, include: str | None) -> bool:
        """Check if a file matches the include glob pattern."""
        if not include:
            return True
        return fnmatch.fnmatch(path.name, include)

    def validate_args(self, args: dict[str, Any]) -> str | None:
        pattern = args.get("pattern")
        if not pattern:
            return "Missing required argument: 'pattern'"
        if not isinstance(pattern, str):
            return "'pattern' must be a string"
        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        pattern_str: str = kwargs["pattern"]
        path_str: str = kwargs.get("path", ".")
        include: str | None = kwargs.get("include")

        resolved = self._resolve_path(path_str)

        if not resolved.exists():
            return ToolOutput(success=False, error=f"Path not found: '{path_str}'")

        try:
            compiled = re.compile(pattern_str)
        except re.error as e:
            return ToolOutput(success=False, error=f"Invalid regex: {e}")

        results: list[str] = []
        files_searched = 0

        try:
            if resolved.is_file():
                # Search single file
                matches = self._search_file(resolved, compiled)
                files_searched = 1
                results.extend(matches)
            else:
                # Walk directory tree
                for root_path, dirs, files in os.walk(resolved):
                    root = Path(root_path)

                    # Filter out ignored directories (modifies dirs in-place to prune walk)
                    dirs[:] = [
                        d for d in dirs
                        if not self._should_skip_dir(d)
                    ]
                    dirs.sort()

                    for fname in sorted(files):
                        if len(results) >= MAX_RESULTS:
                            break

                        fpath = root / fname

                        if self._should_skip_file(fpath):
                            continue
                        if not self._matches_include(fpath, include):
                            continue

                        files_searched += 1
                        matches = self._search_file(fpath, compiled)
                        results.extend(matches)

                        if len(results) >= MAX_RESULTS:
                            break

        except PermissionError as e:
            return ToolOutput(success=False, error=f"Permission denied: {e}")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Search failed: {exc}")

        if not results:
            return ToolOutput(
                success=True,
                result=f"No matches found for pattern '{pattern_str}' in {files_searched} files.",
            )

        truncated = len(results) >= MAX_RESULTS
        output_lines = results[:MAX_RESULTS]

        result_text = "\n".join(output_lines)
        result_text += f"\n\n[{len(output_lines)} matches in {files_searched} files searched]"
        if truncated:
            result_text += f" (results capped at {MAX_RESULTS} — refine your search)"

        return ToolOutput(success=True, result=result_text)

    def _search_file(self, path: Path, pattern: re.Pattern) -> list[str]:
        """Search a single file for the pattern. Returns formatted match lines."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        matches = []
        try:
            rel_path = path.relative_to(self._project_root)
        except ValueError:
            rel_path = path

        for i, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                display_line = line.rstrip()
                if len(display_line) > MAX_LINE_LENGTH:
                    display_line = display_line[:MAX_LINE_LENGTH] + "..."
                matches.append(f"{rel_path}:{i}: {display_line}")

                if len(matches) >= MAX_RESULTS:
                    break

        return matches

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: project root). Absolute or relative.",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py', '*.ts'). Optional.",
                    },
                },
                "required": ["pattern"],
            },
        }
