"""Git tool — structured git operations for the coding assistant.

Provides git status, diff, log, commit, and branch operations
through a single tool with a sub-command interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_OUTPUT_BYTES = 100 * 1024  # 100KB
DEFAULT_TIMEOUT = 30


class GitTool(BaseTool):
    """Perform git operations in the project directory.

    Supports: status, diff, log, commit, branch, add, checkout.
    Safer than running raw git commands through bash because
    it validates inputs and limits dangerous operations.
    """

    name = "git"
    description = (
        "Perform git operations in the project. Supported sub-commands: "
        "'status' (show working tree status), "
        "'diff' (show changes, optionally for a specific file), "
        "'log' (show recent commits), "
        "'commit' (stage and commit changes with a message), "
        "'add' (stage files), "
        "'branch' (list or create branches), "
        "'checkout' (switch branches or restore files). "
        "Use this instead of raw bash git commands for safer, structured output."
    )

    def __init__(self, cwd: Path | None = None) -> None:
        self._cwd = cwd or Path.cwd()

    def validate_args(self, args: dict[str, Any]) -> str | None:
        subcommand = args.get("subcommand")
        if not subcommand:
            return "Missing required argument: 'subcommand'"
        valid = {"status", "diff", "log", "commit", "add", "branch", "checkout", "show"}
        if subcommand not in valid:
            return f"Invalid subcommand '{subcommand}'. Valid: {sorted(valid)}"
        if subcommand == "commit" and not args.get("message"):
            return "Missing required argument 'message' for commit"
        return None

    async def _run_git(self, *cmd_args: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[int, str, str]:
        """Run a git command and return (exit_code, stdout, stderr)."""
        full_cmd = ["git"] + list(cmd_args)

        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            env={**os.environ},
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
        return process.returncode or 0, stdout, stderr

    async def run(self, **kwargs: Any) -> ToolOutput:
        subcommand: str = kwargs["subcommand"]

        try:
            if subcommand == "status":
                code, out, err = await self._run_git("status", "--short", "--branch")
                if code == 0:
                    return ToolOutput(success=True, result=out or "(clean working tree)")
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "diff":
                diff_args = ["diff"]
                file_path = kwargs.get("file")
                staged = kwargs.get("staged", False)
                if staged:
                    diff_args.append("--staged")
                if file_path:
                    diff_args.append("--")
                    diff_args.append(file_path)
                code, out, err = await self._run_git(*diff_args)
                if code == 0:
                    return ToolOutput(success=True, result=out or "(no differences)")
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "log":
                count = min(kwargs.get("count", 10), 50)
                code, out, err = await self._run_git(
                    "log", f"--oneline", f"-{count}", "--decorate"
                )
                if code == 0:
                    return ToolOutput(success=True, result=out or "(no commits)")
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "show":
                ref = kwargs.get("ref", "HEAD")
                code, out, err = await self._run_git("show", "--stat", ref)
                if code == 0:
                    return ToolOutput(success=True, result=out)
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "add":
                files = kwargs.get("files", ["."])
                if isinstance(files, str):
                    files = [files]
                code, out, err = await self._run_git("add", *files)
                if code == 0:
                    return ToolOutput(success=True, result=f"Staged: {', '.join(files)}")
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "commit":
                message = kwargs["message"]
                # Auto-stage all tracked changes before committing
                await self._run_git("add", "-u")
                code, out, err = await self._run_git("commit", "-m", message)
                if code == 0:
                    return ToolOutput(success=True, result=out)
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "branch":
                branch_name = kwargs.get("name")
                if branch_name:
                    code, out, err = await self._run_git("checkout", "-b", branch_name)
                else:
                    code, out, err = await self._run_git("branch", "-a", "--list")
                if code == 0:
                    return ToolOutput(success=True, result=out or "(no branches)")
                return ToolOutput(success=False, error=err or out)

            elif subcommand == "checkout":
                target = kwargs.get("target")
                if not target:
                    return ToolOutput(success=False, error="Missing 'target' for checkout")
                code, out, err = await self._run_git("checkout", target)
                if code == 0:
                    return ToolOutput(success=True, result=out or f"Switched to {target}")
                return ToolOutput(success=False, error=err or out)

            else:
                return ToolOutput(success=False, error=f"Unknown subcommand: {subcommand}")

        except asyncio.TimeoutError:
            return ToolOutput(success=False, error=f"Git command timed out after {DEFAULT_TIMEOUT}s")
        except Exception as exc:
            logger.error("git_error", extra={"error_type": type(exc).__name__, "error": str(exc)})
            return ToolOutput(success=False, error=f"Git command failed: {exc}")

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "subcommand": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "commit", "add", "branch", "checkout", "show"],
                        "description": "The git operation to perform",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (required for 'commit')",
                    },
                    "file": {
                        "type": "string",
                        "description": "File path for diff (optional)",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths for 'add' (default: ['.'])",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged changes only for 'diff' (default: false)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of commits to show for 'log' (default: 10, max: 50)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Branch name for 'branch' (create new branch)",
                    },
                    "target": {
                        "type": "string",
                        "description": "Branch or file to checkout for 'checkout'",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Git ref for 'show' (default: HEAD)",
                    },
                },
                "required": ["subcommand"],
            },
        }
