"""Multi-edit tool — apply multiple edits to a single file in one call.

Like Claude Code's multi_replace_file_content, this tool allows making
multiple non-contiguous edits to the same file efficiently, returning
a combined diff.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MultiEditTool(BaseTool):
    """Apply multiple search-and-replace edits to a single file.

    Each edit is an {old_str, new_str} pair. Edits are applied sequentially
    in the order provided. Each old_str must match exactly one location
    in the file *at the time of its application*.

    Returns a combined unified diff of all changes.
    """

    name = "multi_edit"
    description = (
        "Apply multiple edits to a single file in one call. Each edit is a "
        "{old_str, new_str} pair applied sequentially. Use this when you need "
        "to make several non-adjacent changes to the same file — more efficient "
        "than multiple separate edit_file calls. Each old_str must match exactly "
        "one location in the file."
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

        edits = args.get("edits")
        if not edits:
            return "Missing required argument: 'edits'"
        if not isinstance(edits, list):
            return "'edits' must be a list of {old_str, new_str} objects"
        if len(edits) == 0:
            return "'edits' list cannot be empty"

        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return f"Edit #{i+1} must be an object with 'old_str' and 'new_str'"
            if "old_str" not in edit:
                return f"Edit #{i+1} missing 'old_str'"
            if "new_str" not in edit:
                return f"Edit #{i+1} missing 'new_str'"
            if not isinstance(edit["old_str"], str):
                return f"Edit #{i+1}: 'old_str' must be a string"
            if not isinstance(edit["new_str"], str):
                return f"Edit #{i+1}: 'new_str' must be a string"

        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        path_str: str = kwargs["path"]
        edits: list[dict[str, str]] = kwargs["edits"]

        resolved = self._resolve_path(path_str)

        if not resolved.exists():
            return ToolOutput(success=False, error=f"File not found: '{path_str}'")
        if not resolved.is_file():
            return ToolOutput(success=False, error=f"Not a file: '{path_str}'")

        try:
            original = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to read file: {exc}")

        current_content = original
        applied_count = 0

        for i, edit in enumerate(edits):
            old_str = edit["old_str"]
            new_str = edit["new_str"]

            # Count occurrences
            count = current_content.count(old_str)

            if count == 0:
                return ToolOutput(
                    success=False,
                    error=(
                        f"Edit #{i+1}: old_str not found in '{path_str}' "
                        f"(after applying {applied_count} previous edits). "
                        f"Make sure the string matches exactly. "
                        f"old_str preview: '{old_str[:80]}...'" if len(old_str) > 80
                        else f"old_str: '{old_str}'"
                    ),
                )

            if count > 1:
                return ToolOutput(
                    success=False,
                    error=(
                        f"Edit #{i+1}: old_str matches {count} locations in '{path_str}'. "
                        f"Must match exactly 1. Include more context to make it unique."
                    ),
                )

            current_content = current_content.replace(old_str, new_str, 1)
            applied_count += 1

        # Generate a unified diff
        original_lines = original.splitlines(keepends=True)
        new_lines = current_content.splitlines(keepends=True)
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
            resolved.write_text(current_content, encoding="utf-8")
        except Exception as exc:
            return ToolOutput(success=False, error=f"Failed to write file: {exc}")

        # Summarize changes
        original_lines_count = original.count("\n") + 1
        new_lines_count = current_content.count("\n") + 1
        delta = new_lines_count - original_lines_count

        summary = f"Applied {applied_count} edits to '{path_str}'"
        if delta > 0:
            summary += f" (+{delta} lines)"
        elif delta < 0:
            summary += f" ({delta} lines)"

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
                    "edits": {
                        "type": "array",
                        "description": (
                            "List of edits to apply sequentially. Each edit has 'old_str' "
                            "(exact text to find) and 'new_str' (replacement text)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_str": {
                                    "type": "string",
                                    "description": "Exact string to find and replace (must match exactly 1 location)",
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": "Replacement string",
                                },
                            },
                            "required": ["old_str", "new_str"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        }
