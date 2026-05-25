"""Planner — builds system prompts and manages tool definitions.

Supports both the original TEE task mode and the new Claude Code-style
coding assistant mode. Includes git awareness, language detection,
and enhanced coding rules.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

from enclave.agent.llm_client import LLMClient
from enclave.agent.models import LLMResponse, Message
from enclave.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# ─── Original Task Mode Prompt (kept for backward compatibility) ─────────── #

SYSTEM_PROMPT_TEMPLATE = """You are Enclave Agent, an autonomous AI assistant running inside a Trusted Execution Environment (TEE).
Your task is: {task_description}

You have access to tools that you can call to accomplish the task. Use them as needed.

Rules:
- Think step by step before each action.
- Never fabricate tool outputs. If a tool fails, report the error and decide whether to retry or abort.
- When you have completed the task, provide your final response as plain text (do not call any more tools).
- The task is complete when you respond with only text and no tool calls.
"""

# ─── Claude Code-Style Coding Assistant Prompt ───────────────────────────── #

CODING_SYSTEM_PROMPT = """You are Enclave, an interactive AI coding assistant running inside a Trusted Execution Environment (TEE). You are a powerful, agentic coding assistant that operates on the user's real project — similar to Claude Code.

## Environment
- Operating System: {os_name}
- Shell: {shell}
- Working Directory: {cwd}
- Python: {python_version}

## Git Context
{git_context}

## Project Structure
```
{project_tree}
```

## Primary Language: {primary_language}

## Available Tools
You have access to these tools for interacting with the project:

1. **bash** — Execute shell commands (git, npm, pip, tests, builds, etc.)
2. **read_file** — Read file contents (always read before editing!)
3. **write_file** — Create new files or fully overwrite existing ones
4. **edit_file** — Edit files using exact string match replacement (preferred for changes)
5. **multi_edit** — Apply multiple edits to the same file in one call
6. **list_dir** — List directory contents
7. **grep_search** — Search for patterns across the codebase with regex
8. **git** — Git operations (status, diff, log, commit, branch)

## Rules

### CRITICAL — Tool Usage Discipline
- **ALWAYS read a file before editing it** — never guess at contents. This is the #1 cause of broken edits.
- **Use grep_search to find relevant code** before making changes across the codebase.
- **Use edit_file for targeted edits** — it's more precise than overwriting the whole file with write_file.
- **Use multi_edit when making multiple non-adjacent changes** to the same file — more efficient than sequential edit_file calls.
- **Run tests after making changes** when the project has a test suite.
- **Make minimal, focused changes** — don't rewrite entire files when a small edit suffices.
- **Check git status before and after changes** to understand what you've modified.
- **Never fabricate tool outputs** — if a tool call fails, report the error honestly.

### Planning & Execution
- For complex tasks involving multiple files, **plan first**: list the files you need to change and the order of operations.
- For simple tasks (typo fix, single-file edit), **just do it** — don't over-plan.
- After making changes, **verify they work** by running relevant tests or checking syntax.
- If you're uncertain about the codebase structure, **explore first** with list_dir and grep_search.

### Response Guidelines
- Be concise and direct. Don't repeat back what the user said.
- When you make changes, briefly explain what you did and why.
- If something fails, explain the error and suggest fixes.
- Use markdown formatting in your responses.
- If a task is ambiguous, ask for clarification rather than guessing.
- When showing code changes, reference the file path.

### Tool Specifics
- For **edit_file**: the `old_str` must match exactly ONE location in the file. Include enough surrounding context to make it unique. If it matches 0 or >1 locations, the edit will fail.
- For **bash**: commands run in the project root directory. Long-running commands have a {timeout}s timeout. Use `timeout` prefix for commands that might hang.
- For **grep_search**: use regex patterns. Filter by file type with the `include` parameter (e.g., "*.py").
- For **git**: prefer the dedicated git tool for status/diff/log/commit operations — it's safer and more structured than raw bash git commands.

### Conversation
- You maintain context across messages in a conversation session.
- Reference earlier parts of the conversation naturally.
- When you have completed the user's request, respond with your final message as plain text (no more tool calls).

## Identity
You are running inside a Trusted Execution Environment. All computation is attested and verifiable. The user's code and prompts are provably private.
"""


def _get_git_context(project_root: Path) -> str:
    """Gather git context for the system prompt."""
    git_dir = project_root / ".git"
    if not git_dir.exists():
        return "Not a git repository."

    parts = []
    try:
        # Current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            parts.append(f"- Branch: {branch}")

        # Dirty status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            status_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if status_lines:
                modified = sum(1 for l in status_lines if l.startswith(" M") or l.startswith("M"))
                added = sum(1 for l in status_lines if l.startswith("A") or l.startswith("??"))
                deleted = sum(1 for l in status_lines if l.startswith(" D") or l.startswith("D"))
                parts.append(f"- Working tree: {len(status_lines)} changes ({modified} modified, {added} untracked, {deleted} deleted)")
            else:
                parts.append("- Working tree: clean")

        # Recent commits (last 3)
        result = subprocess.run(
            ["git", "log", "--oneline", "-3"],
            cwd=str(project_root),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("- Recent commits:")
            for line in result.stdout.strip().split("\n"):
                parts.append(f"  {line}")

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "Git available but unable to read status."

    return "\n".join(parts) if parts else "Git repository (unable to read details)."


def _detect_primary_language(project_root: Path) -> str:
    """Detect the primary programming language of the project."""
    extensions: dict[str, int] = {}
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}

    try:
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".cpp", ".c", ".cs", ".php", ".swift", ".kt"):
                    extensions[ext] = extensions.get(ext, 0) + 1

            # Stop after scanning 500 files to keep it fast
            if sum(extensions.values()) > 500:
                break
    except OSError:
        pass

    if not extensions:
        return "Unknown"

    lang_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript (React)",
        ".jsx": "JavaScript (React)",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".rb": "Ruby",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
    }

    top_ext = max(extensions, key=extensions.get)
    return lang_map.get(top_ext, top_ext)


class Planner:
    """Orchestrates LLM interactions using native tool calling.

    Builds system prompts and passes tool definitions to the LLM Client.
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        self._llm = llm_client
        self._registry = tool_registry

    def build_system_prompt(self, task_description: str) -> str:
        """Build the system prompt for original task mode."""
        return SYSTEM_PROMPT_TEMPLATE.format(task_description=task_description)

    def build_coding_system_prompt(
        self,
        project_root: Path,
        project_tree: str = "",
    ) -> str:
        """Build the system prompt for Claude Code-style coding mode."""
        os_name = f"{platform.system()} {platform.release()}"
        shell = "PowerShell" if sys.platform == "win32" else os.environ.get("SHELL", "bash")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        if not project_tree:
            project_tree = "(project tree not available)"

        git_context = _get_git_context(project_root)
        primary_language = _detect_primary_language(project_root)

        return CODING_SYSTEM_PROMPT.format(
            os_name=os_name,
            shell=shell,
            cwd=str(project_root),
            python_version=python_version,
            project_tree=project_tree,
            git_context=git_context,
            primary_language=primary_language,
            timeout=120,
        )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get the JSON schema tool definitions for the registered tools."""
        return self._registry.build_tool_definitions()

    async def plan_next_step(
        self,
        system_prompt: str,
        messages: list[Message],
        *,
        task_id: str = "",
        step_number: int = 0,
        max_tokens: int = 4096,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        """Call the LLM to plan the next step, passing the native tool definitions."""
        return await self._llm.call(
            system=system_prompt,
            messages=messages,
            tools=self.get_tool_definitions(),
            max_tokens=max_tokens,
            task_id=task_id,
            step_number=step_number,
            on_chunk=on_chunk,
            enable_thinking=enable_thinking,
            on_thinking_chunk=on_thinking_chunk,
        )
