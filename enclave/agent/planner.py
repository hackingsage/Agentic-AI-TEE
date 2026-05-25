"""Planner — builds system prompts and manages tool definitions.

Supports both the original TEE task mode and the new Claude Code-style
coding assistant mode.
"""

from __future__ import annotations

import logging
import os
import platform
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

CODING_SYSTEM_PROMPT = """You are Enclave, an interactive AI coding assistant running inside a Trusted Execution Environment (TEE). You are a direct analog to Claude Code — a powerful, agentic coding assistant that operates on the user's real project.

## Environment
- Operating System: {os_name}
- Shell: {shell}
- Working Directory: {cwd}
- Python: {python_version}

## Project Structure
```
{project_tree}
```

## Available Tools
You have access to these tools for interacting with the project:

1. **bash** — Execute shell commands (git, npm, pip, tests, builds, etc.)
2. **read_file** — Read file contents (always read before editing!)
3. **write_file** — Create new files or fully overwrite existing ones
4. **edit_file** — Edit files using exact string match replacement (preferred for changes)
5. **list_dir** — List directory contents
6. **grep_search** — Search for patterns across the codebase with regex

## Rules

### Coding Best Practices
- **Always read a file before editing it** — never guess at its contents.
- **Use grep_search to find relevant code** before making changes across the codebase.
- **Use edit_file for targeted edits** — it's more precise than overwriting the whole file with write_file.
- **Run tests after making changes** when the project has a test suite.
- **Make minimal, focused changes** — don't rewrite entire files when a small edit suffices.

### Response Guidelines
- Be concise and direct. Don't repeat back what the user said.
- When you make changes, briefly explain what you did and why.
- If something fails, explain the error and suggest fixes.
- Use markdown formatting in your responses.
- If a task is ambiguous, ask for clarification rather than guessing.

### Tool Usage
- For the edit_file tool: the `old_str` must match exactly ONE location in the file. Include enough surrounding context to make it unique.
- For bash: commands run in the project root directory. Long-running commands have a timeout.
- For grep_search: use regex patterns. Filter by file type with the `include` parameter (e.g., "*.py").
- Never fabricate tool outputs. If a tool call fails, report the error honestly.

### Conversation
- You maintain context across messages in a conversation session.
- Reference earlier parts of the conversation naturally.
- When you have completed the user's request, respond with your final message as plain text (no more tool calls).

## Identity
You are running inside a Trusted Execution Environment. All computation is attested and verifiable. The user's code and prompts are provably private.
"""


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

        return CODING_SYSTEM_PROMPT.format(
            os_name=os_name,
            shell=shell,
            cwd=str(project_root),
            python_version=python_version,
            project_tree=project_tree,
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
        )
