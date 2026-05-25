"""Edit file tool — targeted search-and-replace editing.

Uses exact string matching to find and replace content in files.
Returns a unified diff of the change.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)


class EditFileTool(BaseTool):
    """Edit files using exact string match replacement.

    Finds `old_str` in the file and replaces it with `new_str`.
    The `old_str` must match exactly one location in the file.
    Returns a unified diff showing the change.
    """

    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match with new content. "
        "The old_str must match exactly ONE location in the file (the tool will error "
        "if it matches zero or multiple locations). Always read the file first to see "
        "its current contents before editing. The tool returns a diff of the change."
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
        old_str = args.get("old_str")
        if old_str is None:
            return "Missing required argument: 'old_str'"
        if not isinstance(old_str, str):
            return "'old_str' must be a string"
        new_str = args.get("new_str")
        if new_str is None:
            return "Missing required argument: 'new_str'"
        if not isinstance(new_str, str):
            return "'new_str' must be a string"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        path_str: str = kwargs["path"]
        old_str: str = kwargs["old_str"]
        new_str: str = kwargs["new_str"]

        resolved = self._resolve_path(path_str)

        if not resolved.exists():
            return ToolOutput(success=False, error=f"File not found: '{path_str}'")
        if not resolved.is_file():
            return ToolOutput(success=False, error=f"Not a file: '{path_str}'")

        try:
            original = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to read file: {exc}")

        # Count occurrences
        count = original.count(old_str)

        if count == 0:
            # Help the user debug — show a snippet of what's in the file
            return ToolOutput(
                success=False,
                error=(
                    f"old_str not found in '{path_str}'. "
                    f"Make sure the string matches exactly (including whitespace and indentation). "
                    f"Use read_file to see the current contents."
                ),
            )

        if count > 1:
            return ToolOutput(
                success=False,
                error=(
                    f"old_str matches {count} locations in '{path_str}'. "
                    f"It must match exactly 1 location. "
                    f"Include more surrounding context in old_str to make it unique."
                ),
            )

        # Perform the replacement
        new_content = original.replace(old_str, new_str, 1)

        # Generate a unified diff
        original_lines = original.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{path_str}",
            tofile=f"b/{path_str}",
            n=3,
        )
        diff_text = "".join(diff)

        # Write the file
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to write file: {exc}")

        # Summarize the change
        old_line_count = old_str.count("\n") + 1
        new_line_count = new_str.count("\n") + 1
        delta = new_line_count - old_line_count

        summary = f"Edited '{path_str}'"
        if delta > 0:
            summary += f" (+{delta} lines)"
        elif delta < 0:
            summary += f" ({delta} lines)"
        else:
            summary += f" ({old_line_count} lines changed)"

        result = f"{summary}\n\n{diff_text}" if diff_text else summary

        return ToolOutput(success=True, result=result)

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (absolute or relative to project root)",
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "The exact string to find and replace. Must match exactly one "
                            "location in the file. Include enough context (surrounding lines) "
                            "to make it unique."
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement string",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        }
