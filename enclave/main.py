"""Enclave entry point — bootstraps and runs the agent service inside the TEE.

This is the main process that runs inside the Nitro Enclave.
It initializes crypto, registers tools, starts the vsock server,
and dispatches tasks as they arrive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from enclave.agent.controller import AgentController
from enclave.agent.llm_client import create_llm_client
from enclave.agent.models import TaskRequest, StepEvent
from enclave.crypto.attestation import MockAttestationProvider, AttestationProvider
from enclave.crypto.keys import EnclaveKeyManager
from enclave.crypto.sealing import SealedSecretStore
from enclave.memory.manager import MemoryManager
from enclave.memory.state_db import TaskStateDB
from enclave.tools.api_call import APICallTool
from enclave.tools.base import ToolRegistry
from enclave.tools.browser_tool import BrowserTool
from enclave.tools.code_executor import CodeExecutor
from enclave.tools.file_ops import FileSystem
from enclave.tools.memory_tool import MemoryTool
from enclave.vsock.protocol import MessageFrame, write_frame
from enclave.vsock.server import VsockServer

# --------------------------------------------------------------------------- #
# Logging — structured, never leaks user data
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("enclave.main")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class EnclaveConfig:
    """Enclave runtime configuration — loaded from env vars."""

    def __init__(self) -> None:
        # Transport
        self.use_vsock = os.getenv("ENCLAVE_USE_VSOCK", "false").lower() == "true"
        self.vsock_port = int(os.getenv("ENCLAVE_VSOCK_PORT", "5000"))
        self.tcp_port = int(os.getenv("ENCLAVE_TCP_PORT", "8765"))
        self.tcp_host = os.getenv("ENCLAVE_TCP_HOST", "127.0.0.1")

        # LLM — supports: "anthropic", "openai", "gemini", "openrouter", "groq", "ollama", "mock"
        self.llm_provider = os.getenv("ENCLAVE_LLM_PROVIDER", "mock")
        self.llm_model = os.getenv("ENCLAVE_LLM_MODEL", "claude-sonnet-4-20250514")
        self.llm_base_url = os.getenv("ENCLAVE_LLM_BASE_URL", "")  # For proxy routing

        # Provider API keys (the active one is selected by llm_provider)
        self.llm_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Workspace
        self.workspace_root = Path(os.getenv("ENCLAVE_WORKSPACE", "/tmp/enclave_workspace"))
        self.db_path = Path(os.getenv("ENCLAVE_DB_PATH", "/tmp/enclave_state.db"))
        self.sealed_path = Path(os.getenv("ENCLAVE_SEALED_PATH", "/tmp/enclave_sealed"))

        # Agent defaults
        self.default_max_steps = int(os.getenv("ENCLAVE_MAX_STEPS", "50"))
        self.default_timeout = float(os.getenv("ENCLAVE_TIMEOUT", "300"))
        self.default_budget = float(os.getenv("ENCLAVE_BUDGET_USD", "5.0"))

        # Tool config
        self.domain_allowlist_str = os.getenv("ENCLAVE_DOMAIN_ALLOWLIST", "")

    @property
    def domain_allowlist(self) -> list[str] | None:
        if not self.domain_allowlist_str:
            return None
        return [d.strip() for d in self.domain_allowlist_str.split(",") if d.strip()]


# --------------------------------------------------------------------------- #
# Service Bootstrap
# --------------------------------------------------------------------------- #

class EnclaveService:
    """Top-level service that wires all components and runs the event loop."""

    def __init__(self, config: EnclaveConfig) -> None:
        self.config = config
        self._shutdown_event = asyncio.Event()

        # -- Crypto --
        self.key_manager = EnclaveKeyManager()
        self.sealed_store = SealedSecretStore(storage_path=config.sealed_path)

        # -- Attestation --
        self.attestation: AttestationProvider = MockAttestationProvider()
        # In production: NitroAttestationProvider()

        # -- Memory --
        self.memory_manager = MemoryManager(
            self.key_manager.keys.master_secret,
        )
        self.state_db = TaskStateDB(db_path=config.db_path)

        # -- Tools --
        self.tool_registry = ToolRegistry()
        self._register_tools()

        # -- LLM Client --
        api_key = {
            "anthropic": config.llm_api_key,
            "openai": config.openai_api_key,
            "gemini": config.google_api_key,
            "openrouter": config.openrouter_api_key,
            "groq": config.groq_api_key,
        }.get(config.llm_provider, "")

        base_url = config.llm_base_url
        if config.llm_provider == "ollama" and not base_url:
            base_url = config.ollama_base_url

        llm = create_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=api_key,
            base_url=base_url,
        )

        # -- Agent Controller --
        self.controller = AgentController(
            llm,
            self.tool_registry,
            default_timeout=config.default_timeout,
        )

        # -- Vsock Server --
        self.server = VsockServer(
            use_vsock=config.use_vsock,
            port=config.vsock_port if config.use_vsock else config.tcp_port,
            host=config.tcp_host,
        )
        self._register_handlers()

    def _register_tools(self) -> None:
        """Register all available tools."""
        config = self.config
        config.workspace_root.mkdir(parents=True, exist_ok=True)

        self.tool_registry.register(CodeExecutor(workspace_dir=config.workspace_root))
        self.tool_registry.register(FileSystem(workspace_dir=config.workspace_root))
        self.tool_registry.register(BrowserTool())
        self.tool_registry.register(APICallTool(domain_allowlist=config.domain_allowlist))
        self.tool_registry.register(MemoryTool())

        logger.info(
            "tools_registered",
            extra={"count": len(self.tool_registry.list_tools())},
        )

    def _register_handlers(self) -> None:
        """Register vsock message handlers."""
        self.server.register_handler("task_request", self._handle_task_request)
        self.server.register_handler("attest", self._handle_attest)
        self.server.register_handler("echo", self._handle_echo)
        self.server.register_handler("status", self._handle_status)
        self.server.register_handler("reconfigure", self._handle_reconfigure)

    # ---- Message Handlers ---- #

    async def _handle_task_request(self, msg: MessageFrame, *, writer: asyncio.StreamWriter | None = None) -> MessageFrame:
        """Handle a task_request message from the host."""
        try:
            payload = msg.payload
            task = TaskRequest(
                task_id=payload.get("task_id", ""),
                user_id=payload.get("user_id", "anonymous"),
                description=payload.get("description", ""),
                max_steps=payload.get("max_steps", self.config.default_max_steps),
                budget_usd=payload.get("budget_usd", self.config.default_budget),
                timeout_seconds=payload.get("timeout_seconds", self.config.default_timeout),
            )

            if not task.description:
                return MessageFrame(
                    msg_type="task_error",
                    payload={"error": "Empty task description"},
                    request_id=msg.request_id,
                )

            logger.info("task_received", extra={"task_id": task.task_id})

            # Record in state DB
            await self.state_db.create_task(task)

            # Setup streaming if writer is available
            stream_task = None
            if writer is not None:
                event_queue = self.controller.enable_streaming()

                async def stream_events_loop():
                    try:
                        while True:
                            event = await event_queue.get()
                            frame = MessageFrame(
                                msg_type="step_event",
                                payload={
                                    "task_id": event.task_id,
                                    "step_number": event.step_number,
                                    "event_type": event.event_type,
                                    "data": event.data,
                                    "timestamp": event.timestamp,
                                },
                                request_id=msg.request_id,
                            )
                            await write_frame(writer, frame)
                            event_queue.task_done()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error streaming task events: {e}")

                stream_task = asyncio.create_task(stream_events_loop())

            try:
                # Run the task
                result = await self.controller.run_task(task)
            finally:
                if stream_task is not None:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Record completion
            await self.state_db.complete_task(result)

            # Record individual steps
            for step in result.steps:
                await self.state_db.record_step(task.task_id, step)

            # Build attestation receipt
            attestation_doc = await self.attestation.get_document(
                user_data=result.attestation_hash.encode() if result.attestation_hash else None,
            )

            receipt = {
                "task_id": result.task_id,
                "task_hash": result.attestation_hash or "",
                "pcrs": {str(k): v.hex() for k, v in attestation_doc.pcrs.items()},
                "signature": self.key_manager.sign_detached(
                    (result.attestation_hash or "").encode()
                ).hex(),
            }

            # Extract the final LLM response text for display in the TUI,
            # scanning backwards through steps to find a non-empty response.
            response_text = ""
            if result.steps:
                for step in reversed(result.steps):
                    if step.llm_response:
                        cleaned = step.llm_response.strip()
                        if cleaned:
                            response_text = cleaned
                            break
                if not response_text:
                    response_text = (result.steps[-1].llm_response or "").strip() or result.summary

            return MessageFrame(
                msg_type="task_result",
                payload={
                    "task_id": result.task_id,
                    "success": result.success,
                    "summary": result.summary,
                    "response": response_text,
                    "total_cost_usd": result.total_cost_usd,
                    "elapsed_seconds": result.elapsed_seconds,
                    "steps_count": len(result.steps),
                    "error": result.error,
                    "attestation": receipt,
                },
                request_id=msg.request_id,
            )
        except Exception as e:
            logger.error(f"Error handling task request: {e}", exc_info=True)
            raise

    async def _handle_attest(self, msg: MessageFrame) -> MessageFrame:
        """Return a fresh attestation document."""
        nonce = msg.payload.get("nonce", "").encode() or None
        doc = await self.attestation.get_document(nonce=nonce)

        return MessageFrame(
            msg_type="attestation",
            payload={
                "pcrs": {str(k): v.hex() for k, v in doc.pcrs.items()},
                "verify_key": self.key_manager.keys.verify_key.encode().hex(),
                "public_key": self.key_manager.keys.public_key.encode().hex(),
            },
            request_id=msg.request_id,
        )

    async def _handle_echo(self, msg: MessageFrame) -> MessageFrame:
        """Echo handler for health checks."""
        return MessageFrame(
            msg_type="echo_response",
            payload=msg.payload,
            request_id=msg.request_id,
        )

    async def _handle_status(self, msg: MessageFrame) -> MessageFrame:
        """Return enclave status."""
        return MessageFrame(
            msg_type="status_response",
            payload={
                "status": "running",
                "tools": self.tool_registry.names,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "use_vsock": self.config.use_vsock,
            },
            request_id=msg.request_id,
        )

    async def _handle_reconfigure(self, msg: MessageFrame) -> MessageFrame:
        """Hot-swap the LLM provider/model without restarting the enclave."""
        payload = msg.payload
        new_provider = payload.get("provider", self.config.llm_provider)
        new_model = payload.get("model", self.config.llm_model)
        api_key = payload.get("api_key", "")
        base_url = payload.get("base_url", "")

        try:
            # Update config
            self.config.llm_provider = new_provider
            self.config.llm_model = new_model

            # Store the key on the config so future reconfigures can fall back
            key_attr = {
                "anthropic": "llm_api_key",
                "openai": "openai_api_key",
                "gemini": "google_api_key",
                "openrouter": "openrouter_api_key",
                "groq": "groq_api_key",
            }.get(new_provider)
            if key_attr and api_key:
                setattr(self.config, key_attr, api_key)

            if new_provider == "ollama" and not base_url:
                base_url = self.config.ollama_base_url

            llm = create_llm_client(
                provider=new_provider,
                model=new_model,
                api_key=api_key,
                base_url=base_url,
            )

            # Swap the LLM client inside the running controller
            self.controller._planner._llm = llm

            logger.info(
                "reconfigure_success",
                extra={"provider": new_provider, "model": new_model},
            )

            return MessageFrame(
                msg_type="reconfigure_response",
                payload={
                    "success": True,
                    "provider": new_provider,
                    "model": new_model,
                },
                request_id=msg.request_id,
            )
        except Exception as exc:
            logger.error(
                "reconfigure_failed",
                extra={"error": str(exc)},
            )
            return MessageFrame(
                msg_type="reconfigure_response",
                payload={
                    "success": False,
                    "error": str(exc),
                },
                request_id=msg.request_id,
            )

    # ---- Lifecycle ---- #

    async def run(self) -> None:
        """Start the enclave service."""
        logger.info(
            "enclave_starting",
            extra={
                "use_vsock": self.config.use_vsock,
                "llm_provider": self.config.llm_provider,
                "tools": self.tool_registry.list_tools(),
            },
        )

        await self.state_db.initialize()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Start server
        server_task = asyncio.create_task(self.server.start())

        try:
            await self._shutdown_event.wait()
        finally:
            logger.info("enclave_shutting_down")
            await self.server.stop()
            await self.state_db.close()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

        logger.info("enclave_stopped")


def main() -> None:
    """Entry point."""
    # Ensure UTF-8 output on Windows terminals (prevents CP1252 crashes)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    config = EnclaveConfig()
    service = EnclaveService(config)
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
