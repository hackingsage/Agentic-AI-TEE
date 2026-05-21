#!/usr/bin/env python3
"""Enclave TUI — Interactive terminal frontend for the TEE-Protected AI Agent.

Boots an embedded enclave service, verifies attestation, and provides
a polished interactive command loop for submitting tasks and inspecting results.

Usage:
    python tui.py
    # or after `pip install -e .`:
    enclave-tui
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

# ─── Enclave imports ────────────────────────────────────────────────────────── #

from enclave.agent.llm_client import (
    MODEL_CATALOG,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_ENV_KEYS,
    get_available_providers,
)
from enclave.main import EnclaveConfig, EnclaveService
from enclave.memory.state_db import TaskStateDB
from enclave.vsock.client import VsockClient
from enclave.vsock.protocol import MessageFrame

# ─── Constants ──────────────────────────────────────────────────────────────── #

VERSION = "0.1.0"
ACCENT = "cyan"
DIM = "dim"
SUCCESS = "green"
ERROR = "red"
WARN = "yellow"

# Persistent config file for API keys and preferred model
CONFIG_DIR = Path.home() / ".enclave"
CONFIG_FILE = CONFIG_DIR / "config.json"

BANNER = r"""
[bold cyan]
  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║     ███████╗███╗   ██╗ ██████╗██╗      █████╗ ██╗   ██╗███████╗  ║
  ║     ██╔════╝████╗  ██║██╔════╝██║     ██╔══██╗██║   ██║██╔════╝  ║
  ║     █████╗  ██╔██╗ ██║██║     ██║     ███████║██║   ██║█████╗    ║
  ║     ██╔══╝  ██║╚██╗██║██║     ██║     ██╔══██║╚██╗ ██╔╝██╔══╝    ║
  ║     ███████╗██║ ╚████║╚██████╗███████╗██║  ██║ ╚████╔╝ ███████╗  ║
  ║     ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝  ║
  ║                                                               ║
  ║        [bold white]TEE-Protected AI Agent Platform[/bold white]                      ║
  ║        [dim white]Your code, your data, your prompts — provably private.[/dim white] ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
[/bold cyan]"""

HELP_TEXT = """
[bold cyan]Available Commands:[/bold cyan]

  [bold white]task [dim]<description>[/dim][/bold white]    Submit a task to the secure enclave
  [bold white]model[/bold white]                Switch LLM provider / model
  [bold white]status[/bold white]               Show enclave status (tools, LLM, uptime)
  [bold white]attest[/bold white]               Fetch and display attestation document
  [bold white]history[/bold white]              List all executed tasks
  [bold white]inspect [dim]<task_id>[/dim][/bold white]   Show detailed breakdown of a task
  [bold white]help[/bold white]                 Show this help message
  [bold white]clear[/bold white]                Clear the terminal
  [bold white]exit[/bold white]                 Graceful shutdown

[dim]You can also just type a prompt directly to submit it as a task.[/dim]
"""


# ─── TUI Application ───────────────────────────────────────────────────────── #


class EnclaveTUI:
    """Interactive terminal frontend for the Enclave platform."""

    def __init__(self) -> None:
        self.console = Console()
        self.service: EnclaveService | None = None
        self.client: VsockClient | None = None
        self.config: EnclaveConfig | None = None
        self.server_task: asyncio.Task | None = None
        self.port: int = 0
        self.boot_time: float = 0.0
        self._task_counter: int = 0
        self._task_history: list[dict[str, Any]] = []

    # ── Persistence ──────────────────────────────────────────────────────── #

    def _load_saved_config(self) -> dict[str, Any]:
        """Load saved config (API keys, preferred model) from disk."""
        if not CONFIG_FILE.exists():
            return {}
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_config(
        self, provider: str, model: str, api_keys: dict[str, str] | None = None
    ) -> None:
        """Save current provider, model, and API keys to disk."""
        existing = self._load_saved_config()
        existing["preferred_provider"] = provider
        existing["preferred_model"] = model

        # Merge API keys — keep previously saved keys for other providers
        saved_keys = existing.get("api_keys", {})
        if api_keys:
            saved_keys.update(api_keys)
        existing["api_keys"] = saved_keys

        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(
                json.dumps(existing, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self.console.print(
                f"  [yellow]Warning: Could not save config: {exc}[/yellow]"
            )

    def _apply_saved_config(self, saved: dict[str, Any]) -> None:
        """Apply saved config to the current EnclaveConfig and environment."""
        # Restore API keys into env vars and config
        api_keys = saved.get("api_keys", {})
        key_to_attr = {
            "ANTHROPIC_API_KEY": "llm_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "GOOGLE_API_KEY": "google_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "GROQ_API_KEY": "groq_api_key",
        }
        for env_var, value in api_keys.items():
            if value and not os.getenv(env_var):
                os.environ[env_var] = value
            attr = key_to_attr.get(env_var)
            if attr and value:
                setattr(self.config, attr, value)

        # Restore preferred provider and model
        provider = saved.get("preferred_provider", "")
        model = saved.get("preferred_model", "")
        if provider and model:
            self.config.llm_provider = provider
            self.config.llm_model = model

    # ── Lifecycle ───────────────────────────────────────────────────────── #

    async def boot(self) -> None:
        """Boot the embedded enclave service."""
        self.console.print(BANNER)
        self.console.print(
            f"  [dim]v{VERSION}  •  Python {sys.version.split()[0]}  •  "
            f"PID {os.getpid()}[/dim]\n"
        )

        with self.console.status(
            "[bold cyan]  Initializing enclave...[/bold cyan]",
            spinner="dots",
        ) as status:
            # ── Configure ──
            self.config = EnclaveConfig()
            self.config.use_vsock = False
            self.config.tcp_host = "127.0.0.1"

            # Load saved preferences (API keys, preferred model)
            saved = self._load_saved_config()
            if saved:
                self._apply_saved_config(saved)
            elif not os.getenv("ENCLAVE_LLM_PROVIDER"):
                self.config.llm_provider = "mock"

            status.update("[bold cyan]  Generating enclave crypto keys...[/bold cyan]")
            self.service = EnclaveService(self.config)
            self.service.state_db = TaskStateDB()  # in-memory for TUI
            await self.service.state_db.initialize()

            # ── Find free port ──
            status.update("[bold cyan]  Binding vsock transport...[/bold cyan]")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
            sock.close()

            self.config.tcp_port = self.port
            self.service.server._port = self.port

            # ── Start server ──
            status.update("[bold cyan]  Starting enclave server...[/bold cyan]")
            self.server_task = asyncio.create_task(self.service.server.start())
            await asyncio.sleep(0.4)

            # ── Create client ──
            self.client = VsockClient(
                use_vsock=False,
                host="127.0.0.1",
                port=self.port,
            )

            self.boot_time = time.time()

        # ── Show boot summary ──
        provider_label = _format_provider_label(
            self.config.llm_provider, self.config.llm_model
        )

        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column(style="bold cyan", width=18)
        info_table.add_column()
        info_table.add_row("Transport", f"TCP :{self.port}")
        info_table.add_row("LLM Provider", provider_label)
        info_table.add_row("Tools", ", ".join(self.service.tool_registry.names))
        info_table.add_row("Workspace", str(self.config.workspace_root))

        self.console.print(
            Panel(
                info_table,
                title="[bold green]✓ Enclave Online[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # ── Auto-attest ──
        await self._show_attestation()

        self.console.print(
            "\n  [dim]Type [bold white]help[/bold white] for commands, "
            "or just type a prompt to submit a task.[/dim]\n"
        )

    async def shutdown(self) -> None:
        """Graceful shutdown with hard timeout to prevent hangs."""
        self.console.print()
        with self.console.status(
            "[bold cyan]  Shutting down enclave...[/bold cyan]", spinner="dots"
        ):
            try:
                await asyncio.wait_for(self._teardown(), timeout=3.0)
            except asyncio.TimeoutError:
                self.console.print(
                    "  [yellow]Shutdown timed out — forcing exit.[/yellow]"
                )
            except Exception:
                pass  # swallow errors during teardown

        self.console.print(
            Panel(
                "[bold green]Enclave shut down cleanly.[/bold green]\n"
                "[dim]All in-memory data has been wiped. No trace remains.[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )

    async def _teardown(self) -> None:
        """Internal teardown logic — separated so shutdown() can timeout-wrap it."""
        if self.service:
            await self.service.server.stop()
            await self.service.state_db.close()
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        await asyncio.sleep(0.05)

    # ── Command Dispatch ────────────────────────────────────────────────── #

    async def run_loop(self) -> None:
        """Main interactive command loop."""
        while True:
            try:
                raw = Prompt.ask("\n [bold cyan]enclave[/bold cyan]")
                raw = raw.strip()

                if not raw:
                    continue

                parts = raw.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ("exit", "quit", "q"):
                    break
                elif command == "help":
                    self._show_help()
                elif command == "clear":
                    self.console.clear()
                elif command == "status":
                    await self._show_status()
                elif command == "attest":
                    await self._show_attestation()
                elif command == "model":
                    await self._show_model_selector()
                elif command == "history":
                    self._show_history()
                elif command == "inspect":
                    self._show_inspect(args)
                elif command == "task":
                    if not args:
                        self.console.print(
                            "  [red]Usage:[/red] task <description>"
                        )
                    else:
                        await self._run_task(args)
                else:
                    # Treat the entire input as a task description
                    await self._run_task(raw)

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as exc:
                self.console.print(f"\n  [red]Error:[/red] {exc}")
                self.console.print(f"  [dim]{traceback.format_exc()}[/dim]")

    # ── Commands ────────────────────────────────────────────────────────── #

    def _show_help(self) -> None:
        """Display help text."""
        self.console.print(HELP_TEXT)

    async def _show_status(self) -> None:
        """Fetch and display enclave status."""
        with self.console.status("  [cyan]Querying enclave...[/cyan]", spinner="dots"):
            response = await self.client.send(
                MessageFrame(msg_type="status", payload={})
            )

        payload = response.payload
        uptime = time.time() - self.boot_time

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", width=18)
        table.add_column()
        table.add_row("Status", f"[bold green]{payload.get('status', 'unknown')}[/bold green]")
        table.add_row("Uptime", _format_duration(uptime))
        table.add_row("LLM Provider", payload.get("llm_provider", "unknown"))
        table.add_row("Tools", ", ".join(payload.get("tools", [])))
        table.add_row("Transport", "vsock" if payload.get("use_vsock") else f"TCP :{self.port}")
        table.add_row("Tasks Executed", str(len(self._task_history)))

        self.console.print(
            Panel(table, title="[bold cyan]Enclave Status[/bold cyan]", border_style="cyan", padding=(1, 2))
        )

    async def _show_attestation(self) -> None:
        """Fetch and display an attestation document."""
        with self.console.status("  [cyan]Requesting attestation...[/cyan]", spinner="dots"):
            response = await self.client.send(
                MessageFrame(msg_type="attest", payload={"nonce": f"tui-{time.time()}"})
            )

        payload = response.payload
        pcrs = payload.get("pcrs", {})

        # PCR table
        pcr_table = Table(
            title="Platform Configuration Registers (PCRs)",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            title_style="bold white",
        )
        pcr_table.add_column("Index", style="bold cyan", width=6, justify="center")
        pcr_table.add_column("Measurement", style="white")
        pcr_table.add_column("Description", style="dim")

        pcr_descriptions = {
            "0": "Enclave image hash",
            "1": "Kernel measurement",
            "2": "Application code hash",
        }
        for idx, value in sorted(pcrs.items()):
            truncated = value[:24] + "..." + value[-12:] if len(value) > 40 else value
            pcr_table.add_row(
                f"PCR[{idx}]",
                truncated,
                pcr_descriptions.get(idx, ""),
            )

        # Keys
        keys_text = Text()
        verify_key = payload.get("verify_key", "N/A")
        public_key = payload.get("public_key", "N/A")
        keys_text.append("  Verify Key: ", style="bold cyan")
        keys_text.append(f"{verify_key[:16]}...{verify_key[-8:]}\n", style="white")
        keys_text.append("  Public Key: ", style="bold cyan")
        keys_text.append(f"{public_key[:16]}...{public_key[-8:]}", style="white")

        content = Text()
        content.append("")

        self.console.print(
            Panel(
                Columns([pcr_table], padding=(0, 0)),
                title="[bold green]🔐 Attestation Document[/bold green]",
                subtitle="[dim]Cryptographic proof of enclave integrity[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
        self.console.print(keys_text)

    async def _run_task(self, description: str) -> None:
        """Submit a task to the enclave and display results with live updates."""
        self._task_counter += 1
        task_id = f"tui_{self._task_counter:04d}"

        self.console.print()
        self.console.print(
            Rule(f"[bold cyan]Task {task_id}[/bold cyan]", style="cyan")
        )
        self.console.print(f"  [dim]Prompt:[/dim] {description}\n")

        # ── Execute with live status ──
        start = time.time()
        result_payload: dict[str, Any] = {}

        with self.console.status(
            "  [bold cyan]⚡ Executing inside secure enclave...[/bold cyan]",
            spinner="dots",
        ):
            response = await self.client.send(
                MessageFrame(
                    msg_type="task_request",
                    payload={
                        "task_id": task_id,
                        "user_id": "tui_user",
                        "description": description,
                        "max_steps": self.config.default_max_steps,
                        "budget_usd": self.config.default_budget,
                    },
                ),
                timeout=300.0,
            )
            result_payload = response.payload

        elapsed = time.time() - start
        success = result_payload.get("success", False)

        # Store in history
        history_entry = {
            "task_id": task_id,
            "description": description,
            "success": success,
            "summary": result_payload.get("summary", ""),
            "cost_usd": result_payload.get("total_cost_usd", 0.0),
            "elapsed": elapsed,
            "steps": result_payload.get("steps_count", 0),
            "attestation": result_payload.get("attestation", {}),
            "error": result_payload.get("error"),
            "timestamp": time.time(),
        }
        self._task_history.append(history_entry)

        # ── Display Result ──
        if success:
            self._render_task_success(result_payload, elapsed)
        else:
            self._render_task_failure(result_payload, elapsed)

        # ── Display Attestation Receipt ──
        attestation = result_payload.get("attestation", {})
        if attestation:
            self._render_attestation_receipt(attestation)

    def _render_task_success(self, payload: dict, elapsed: float) -> None:
        """Render a successful task result with full response."""
        summary = payload.get("summary", "Task completed.")
        response_text = payload.get("response", "")
        cost = payload.get("total_cost_usd", 0.0)
        steps = payload.get("steps_count", 0)

        # Show the full LLM response as rich Markdown if available
        if response_text:
            self.console.print(
                Panel(
                    Markdown(response_text),
                    title="[bold green]\u2713 Response[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # Show summary + metadata in a compact table below
        meta_table = Table(show_header=False, box=None, padding=(0, 2))
        meta_table.add_column(style="bold white", width=16)
        meta_table.add_column()
        meta_table.add_row("Summary", summary)
        meta_table.add_row("Steps", str(steps))
        meta_table.add_row("Cost", f"${cost:.6f}")
        meta_table.add_row("Elapsed", f"{elapsed:.2f}s")

        self.console.print(
            Panel(
                meta_table,
                title="[bold cyan]Task Details[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def _render_task_failure(self, payload: dict, elapsed: float) -> None:
        """Render a failed task result."""
        error = payload.get("error", "Unknown error")
        summary = payload.get("summary", "Task failed.")
        steps = payload.get("steps_count", 0)

        result_table = Table(show_header=False, box=None, padding=(0, 2))
        result_table.add_column(style="bold white", width=16)
        result_table.add_column()
        result_table.add_row("Error", f"[red]{error}[/red]")
        result_table.add_row("Summary", summary)
        result_table.add_row("Steps", str(steps))
        result_table.add_row("Elapsed", f"{elapsed:.2f}s")

        self.console.print(
            Panel(
                result_table,
                title="[bold red]✗ Task Failed[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )

    def _render_attestation_receipt(self, attestation: dict) -> None:
        """Render the cryptographic attestation receipt for a task."""
        task_hash = attestation.get("task_hash", "")
        signature = attestation.get("signature", "")
        pcrs = attestation.get("pcrs", {})

        receipt_content = Text()
        receipt_content.append("  Task Hash      ", style="bold cyan")
        receipt_content.append(f"{task_hash[:32]}...\n", style="white")

        receipt_content.append("  Signature      ", style="bold cyan")
        sig_display = f"{signature[:32]}...{signature[-16:]}" if len(signature) > 52 else signature
        receipt_content.append(f"{sig_display}\n", style="white")

        for idx, value in sorted(pcrs.items()):
            label = f"  PCR[{idx}]" + " " * (14 - len(f"PCR[{idx}]"))
            receipt_content.append(label, style="bold cyan")
            truncated = f"{value[:24]}...{value[-12:]}" if len(value) > 40 else value
            receipt_content.append(f"{truncated}\n", style="dim white")

        receipt_content.append("\n  ", style="")
        receipt_content.append(
            "This receipt cryptographically proves the task was executed\n"
            "  inside an attested enclave and the output was not tampered with.",
            style="dim italic",
        )

        self.console.print(
            Panel(
                receipt_content,
                title="[bold yellow]🔏 Attestation Receipt[/bold yellow]",
                border_style="yellow",
                padding=(1, 1),
            )
        )

    def _show_history(self) -> None:
        """Display a table of all executed tasks."""
        if not self._task_history:
            self.console.print("  [dim]No tasks have been executed yet.[/dim]")
            return

        table = Table(
            title="Task History",
            box=box.ROUNDED,
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("ID", style="bold white", width=10)
        table.add_column("Description", style="white", max_width=40, overflow="ellipsis")
        table.add_column("Status", justify="center", width=10)
        table.add_column("Steps", justify="center", width=6)
        table.add_column("Cost", justify="right", width=12)
        table.add_column("Time", justify="right", width=8)

        for entry in reversed(self._task_history):
            status = (
                "[bold green]✓ OK[/bold green]"
                if entry["success"]
                else "[bold red]✗ FAIL[/bold red]"
            )
            table.add_row(
                entry["task_id"],
                entry["description"],
                status,
                str(entry["steps"]),
                f"${entry['cost_usd']:.6f}",
                f"{entry['elapsed']:.2f}s",
            )

        self.console.print(Panel(table, border_style="cyan", padding=(1, 2)))

    def _show_inspect(self, task_id: str) -> None:
        """Show detailed breakdown of a specific task."""
        task_id = task_id.strip()
        if not task_id:
            self.console.print("  [red]Usage:[/red] inspect <task_id>")
            return

        entry = None
        for t in self._task_history:
            if t["task_id"] == task_id:
                entry = t
                break

        if not entry:
            self.console.print(f"  [red]Task '{task_id}' not found.[/red]")
            self.console.print("  [dim]Use 'history' to see available task IDs.[/dim]")
            return

        # ── Detail table ──
        detail_table = Table(show_header=False, box=None, padding=(0, 2))
        detail_table.add_column(style="bold cyan", width=18)
        detail_table.add_column()
        detail_table.add_row("Task ID", entry["task_id"])
        detail_table.add_row("Description", entry["description"])
        detail_table.add_row(
            "Status",
            "[bold green]Completed[/bold green]" if entry["success"] else "[bold red]Failed[/bold red]",
        )
        detail_table.add_row("Summary", entry.get("summary", "—"))
        detail_table.add_row("Steps", str(entry["steps"]))
        detail_table.add_row("Cost", f"${entry['cost_usd']:.6f}")
        detail_table.add_row("Elapsed", f"{entry['elapsed']:.2f}s")
        if entry.get("error"):
            detail_table.add_row("Error", f"[red]{entry['error']}[/red]")

        self.console.print(
            Panel(
                detail_table,
                title=f"[bold cyan]Task Detail — {task_id}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # ── Attestation receipt ──
        attestation = entry.get("attestation", {})
        if attestation:
            self._render_attestation_receipt(attestation)

    async def _show_model_selector(self) -> None:
        """Interactive model/provider selection menu."""
        available = get_available_providers()

        # Build a flat numbered list of all models
        options: list[tuple[str, Any]] = []  # (provider, ModelInfo)
        self.console.print()

        table = Table(
            title="Available Models",
            box=box.ROUNDED,
            title_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", style="bold white", width=4, justify="right")
        table.add_column("Provider", style="bold cyan", width=18)
        table.add_column("Model", style="white", width=24)
        table.add_column("Description", style="dim", max_width=40)
        table.add_column("Status", justify="center", width=14)

        idx = 1
        for provider, models in MODEL_CATALOG.items():
            if provider == "mock":
                continue  # skip mock from the UI list
            is_available = provider in available
            env_key = PROVIDER_ENV_KEYS.get(provider, "")

            for model_info in models:
                status_text = (
                    "[bold green]● Ready[/bold green]"
                    if is_available
                    else f"[dim red]○ Set {env_key}[/dim red]"
                )
                if provider == "ollama":
                    status_text = "[yellow]● Local[/yellow]"

                table.add_row(
                    str(idx),
                    PROVIDER_DISPLAY_NAMES.get(provider, provider),
                    model_info.name,
                    model_info.description,
                    status_text,
                )
                options.append((provider, model_info))
                idx += 1

        # Add custom option
        table.add_row(
            str(idx),
            "Custom",
            "Custom Model",
            "Enter any provider and model ID manually",
            "[bold green]● Ready[/bold green]",
        )
        options.append(("custom", None))
        idx += 1

        # Add mock at the end
        table.add_row(
            str(idx),
            "Mock",
            "Mock LLM",
            "Testing only",
            "[bold green]● Ready[/bold green]",
        )
        mock_info = MODEL_CATALOG["mock"][0]
        options.append(("mock", mock_info))

        # Current model indicator
        current = (
            f"  [dim]Current: [bold white]"
            f"{PROVIDER_DISPLAY_NAMES.get(self.config.llm_provider, self.config.llm_provider)}"
            f"[/bold white] / [bold white]{self.config.llm_model}[/bold white][/dim]"
        )
        self.console.print(Panel(table, border_style="cyan", padding=(1, 2)))
        self.console.print(current)

        # Prompt for selection
        choice = Prompt.ask(
            "\n  [cyan]Select model number (or 'cancel')[/cyan]",
            default="cancel",
        )

        if choice.lower() in ("cancel", "c", ""):
            self.console.print("  [dim]Cancelled.[/dim]")
            return

        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(options):
                raise ValueError
        except ValueError:
            self.console.print("  [red]Invalid selection.[/red]")
            return

        provider, model_info = options[choice_idx]

        model_id = ""
        model_name = ""

        if provider == "custom":
            # Prompt for custom provider
            provider_choice = Prompt.ask(
                "  [cyan]Enter provider (anthropic, openai, gemini, openrouter, ollama)[/cyan]"
            ).strip().lower()

            if provider_choice not in PROVIDER_ENV_KEYS:
                self.console.print(f"  [red]Unsupported or unrecognized provider: {provider_choice}[/red]")
                return

            # Prompt for custom model ID
            model_id = Prompt.ask(
                "  [cyan]Enter custom model ID (e.g. deepseek/deepseek-v4-flash:free)[/cyan]"
            ).strip()

            if not model_id:
                self.console.print(f"  [red]Model ID cannot be empty.[/red]")
                return

            provider = provider_choice
            model_name = model_id
        else:
            model_id = model_info.id
            model_name = model_info.name

        # ── API Key prompt (for providers that need one) ──
        env_key = PROVIDER_ENV_KEYS.get(provider, "")
        api_key = ""

        if provider not in ("mock", "ollama") and env_key:
            existing_key = os.getenv(env_key, "")
            if existing_key:
                masked = existing_key[:4] + "•" * (len(existing_key) - 8) + existing_key[-4:]
                self.console.print(
                    f"\n  [dim]Current API key ([bold]{env_key}[/bold]): {masked}[/dim]"
                )
                key_input = Prompt.ask(
                    f"  [cyan]Enter API key for {PROVIDER_DISPLAY_NAMES.get(provider, provider)} "
                    f"(Enter to keep current)[/cyan]",
                    default="",
                    password=True,
                )
            else:
                self.console.print(
                    f"\n  [yellow]No API key found for [bold]{env_key}[/bold].[/yellow]"
                )
                key_input = Prompt.ask(
                    f"  [cyan]Enter API key for {PROVIDER_DISPLAY_NAMES.get(provider, provider)}[/cyan]",
                    default="",
                    password=True,
                )

            api_key = key_input.strip() if key_input.strip() else existing_key

            if not api_key:
                self.console.print(
                    f"  [red]No API key provided. Cannot switch to {PROVIDER_DISPLAY_NAMES.get(provider, provider)}.[/red]"
                )
                return

            # Persist the key in the environment so downstream components pick it up
            os.environ[env_key] = api_key

            # Also update the in-memory config so EnclaveService reads the new key
            _CONFIG_KEY_ATTR = {
                "anthropic": "llm_api_key",
                "openai": "openai_api_key",
                "gemini": "google_api_key",
                "openrouter": "openrouter_api_key",
                "groq": "groq_api_key",
            }
            attr = _CONFIG_KEY_ATTR.get(provider)
            if attr:
                setattr(self.config, attr, api_key)

        # If same as current, skip
        if provider == self.config.llm_provider and model_id == self.config.llm_model:
            self.console.print("  [dim]Already using this model.[/dim]")
            return

        # Restart enclave with new config
        self.console.print(
            f"\n  [cyan]Switching to[/cyan] [bold white]{model_name}[/bold white] "
            f"[cyan]({PROVIDER_DISPLAY_NAMES.get(provider, provider)})...[/cyan]"
        )
        await self._restart_enclave(provider, model_id)

        # Save the new preferences to disk
        keys_to_save = {}
        if api_key and env_key:
            keys_to_save[env_key] = api_key
        self._save_config(provider, model_id, keys_to_save)

    async def _restart_enclave(self, new_provider: str, new_model: str) -> None:
        """Send a reconfigure message to the enclave over the socket protocol.

        This keeps the host/enclave boundary clean — the TUI never
        reaches into enclave memory directly.
        """
        with self.console.status(
            "  [bold cyan]Reconfiguring enclave LLM...[/bold cyan]",
            spinner="dots",
        ):
            # Resolve the API key to forward to the enclave
            api_key = {
                "anthropic": self.config.llm_api_key,
                "openai": self.config.openai_api_key,
                "gemini": self.config.google_api_key,
                "openrouter": self.config.openrouter_api_key,
                "groq": self.config.groq_api_key,
            }.get(new_provider, "")

            base_url = self.config.llm_base_url
            if new_provider == "ollama" and not base_url:
                base_url = self.config.ollama_base_url

            # Send reconfigure frame to the enclave server
            response = await self.client.send(
                MessageFrame(
                    msg_type="reconfigure",
                    payload={
                        "provider": new_provider,
                        "model": new_model,
                        "api_key": api_key,
                        "base_url": base_url or "",
                    },
                ),
                timeout=10.0,
            )

            result = response.payload
            if not result.get("success"):
                error = result.get("error", "Unknown error")
                self.console.print(
                    f"  [red]Reconfigure failed:[/red] {error}"
                )
                return

            # Update the local config to stay in sync
            self.config.llm_provider = new_provider
            self.config.llm_model = new_model

        provider_label = _format_provider_label(new_provider, new_model)

        self.console.print(
            Panel(
                f"  LLM Provider    {provider_label}\n"
                f"  Transport       TCP :{self.port}",
                title="[bold green]✓ Enclave Reconfigured[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


# ─── Helpers ────────────────────────────────────────────────────────────────── #


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _format_provider_label(provider: str, model: str) -> str:
    """Format a styled provider/model label for display."""
    display = PROVIDER_DISPLAY_NAMES.get(provider, provider)
    if provider == "mock":
        return "[bold yellow]Mock LLM[/bold yellow] [dim](use 'model' command to switch)[/dim]"
    return f"[bold green]{display}[/bold green] ({model})"


# ─── Entry Point ────────────────────────────────────────────────────────────── #


async def _async_main() -> None:
    """Async entry point."""
    tui = EnclaveTUI()

    try:
        await tui.boot()
        await tui.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        await tui.shutdown()


def main() -> None:
    """Sync entry point."""
    # Ensure UTF-8 output on Windows terminals (prevents CP1252 crashes)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Suppress enclave logging noise in TUI mode
    import logging

    logging.getLogger("enclave").setLevel(logging.WARNING)

    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
