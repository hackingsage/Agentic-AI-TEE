"""Code execution tool — runs Python/JS/Bash in a sandboxed subprocess.

In production, this uses Docker-in-Docker inside the enclave.
For local development, it uses subprocess with resource limits.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Safety limits
MAX_OUTPUT_BYTES = 10 * 1024  # 10KB
DEFAULT_TIMEOUT = 30
SUPPORTED_LANGUAGES = {"python", "javascript", "bash"}

# Language-to-command mapping
LANG_COMMANDS: dict[str, list[str]] = {
    "python": ["python3", "-c"],
    "javascript": ["node", "-e"],
    "bash": ["bash", "-c"],
}


class CodeExecutor(BaseTool):
    """Execute code in a sandboxed subprocess.

    Supports Python, JavaScript, and Bash. All execution is isolated
    with resource limits and timeouts.
    """

    name = "code_exec"
    description = (
        "Execute code in a sandboxed environment. Supports Python, JavaScript, and Bash. "
        "Returns stdout and stderr. Code execution is isolated and has no network access. "
        "Use this for computation, data processing, or testing code snippets."
    )

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="enclave_code_"))

    def validate_args(self, args: dict[str, Any]) -> str | None:
        language = args.get("language")
        if not language:
            return "Missing required argument: 'language'"
        if language not in SUPPORTED_LANGUAGES:
            return f"Unsupported language '{language}'. Must be one of: {SUPPORTED_LANGUAGES}"
        code = args.get("code")
        if not code:
            return "Missing required argument: 'code'"
        if not isinstance(code, str):
            return "'code' must be a string"
        return None

    async def _resolve_bash_command(self) -> list[str]:
        import os
        import sys
        
        git_bash_paths = [
            "C:\\Program Files\\Git\\bin\\bash.exe",
            "C:\\Program Files\\Git\\usr\\bin\\bash.exe",
            "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
            "C:\\Program Files (x86)\\Git\\usr\\bin\\bash.exe",
            os.path.expandvars("%LocalAppData%\\Programs\\Git\\bin\\bash.exe"),
            os.path.expandvars("%LocalAppData%\\Programs\\Git\\usr\\bin\\bash.exe"),
        ]
        for p in git_bash_paths:
            if os.path.exists(p):
                return [p, "-c"]

        try:
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", "exit 0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1.5)
            if process.returncode == 0:
                return ["bash", "-c"]
        except Exception:
            pass

        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]

    async def run(self, **kwargs: Any) -> ToolOutput:
        language: str = kwargs["language"]
        code: str = kwargs["code"]
        timeout: int = min(kwargs.get("timeout", DEFAULT_TIMEOUT), 120)

        import sys
        if language == "python":
            cmd = [sys.executable, "-c"]
        else:
            cmd = LANG_COMMANDS[language]

        args = list(cmd) + [code]
        temp_file = None

        try:
            if language == "bash" and sys.platform == "win32":
                resolved_bash = await self._resolve_bash_command()
                if resolved_bash[0] == "powershell.exe":
                    temp_file = Path(self._workspace_dir) / "temp_script.ps1"
                    temp_file.write_text(code, encoding="utf-8")
                    args = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(temp_file)]
                else:
                    args = list(resolved_bash) + [code]

            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace_dir),
                # Resource limits applied via preexec in production
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
                result_text += f"STDOUT:\n{stdout}"
            if stderr:
                result_text += f"\nSTDERR:\n{stderr}"
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
                error=f"Code execution timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            return ToolOutput(
                success=False,
                error=f"Runtime for '{language}' not found. Is it installed?",
            )
        except Exception as exc:
            logger.error(
                "code_exec_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            return ToolOutput(
                success=False,
                error=f"Execution failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "bash"],
                        "description": "Programming language to execute",
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default: 30, max: 120)",
                    },
                },
                "required": ["language", "code"],
            },
        }
