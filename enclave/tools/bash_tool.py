"""Bash tool — execute shell commands in the project directory.

Runs commands via subprocess in the project CWD.
On Windows uses PowerShell; on Linux/Mac uses bash.
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
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 300


class BashTool(BaseTool):
    """Execute shell commands in the project directory.

    Runs commands directly in the user's project — no sandboxing.
    On Windows, uses PowerShell. On Unix, uses bash.
    """

    name = "bash"
    description = (
        "Execute a shell command in the project directory. "
        "Use this for running tests, installing packages, git operations, "
        "building projects, or any command-line task. "
        "Commands run in the project root directory. "
        "Output (stdout + stderr) is captured and returned."
    )

    def __init__(self, cwd: Path | None = None) -> None:
        self._cwd = cwd or Path.cwd()

    def validate_args(self, args: dict[str, Any]) -> str | None:
        command = args.get("command")
        if not command:
            return "Missing required argument: 'command'"
        if not isinstance(command, str):
            return "'command' must be a string"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        command: str = kwargs["command"]
        timeout: int = min(kwargs.get("timeout", DEFAULT_TIMEOUT), MAX_TIMEOUT)

        try:
            if sys.platform == "win32":
                # Use PowerShell on Windows
                args = [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ]
            else:
                args = ["bash", "-c", command]

            process = await asyncio.create_subprocess_exec(
                *args,
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

            exit_code = process.returncode

            result_text = ""
            if stdout:
                result_text += stdout
            if stderr:
                if result_text:
                    result_text += "\n"
                result_text += f"STDERR:\n{stderr}"
            if not result_text:
                result_text = "(no output)"

            result_text += f"\n\nExit code: {exit_code}"

            return ToolOutput(
                success=(exit_code == 0),
                result=result_text,
                error=stderr if exit_code != 0 else None,
            )

        except asyncio.TimeoutError:
            return ToolOutput(
                success=False,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as exc:
            logger.error(
                "bash_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            return ToolOutput(
                success=False,
                error=f"Command failed: {type(exc).__name__}: {exc}",
            )

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default: 120, max: 300)",
                    },
                },
                "required": ["command"],
            },
        }
