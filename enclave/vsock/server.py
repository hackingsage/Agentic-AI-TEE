"""Vsock server — async server for enclave-side communication.

Supports pluggable transport: AF_VSOCK (production) or TCP (local dev).
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, Callable, Coroutine

from enclave.vsock.protocol import MessageFrame, read_frame, write_frame

logger = logging.getLogger(__name__)

# Vsock CID for "any" — used inside the enclave
VMADDR_CID_ANY = 0xFFFFFFFF  # VMADDR_CID_ANY
DEFAULT_VSOCK_PORT = 5000
DEFAULT_TCP_PORT = 8765


class VsockServer:
    """Async server that listens for framed messages.

    In production, listens on AF_VSOCK. For local development,
    falls back to TCP on localhost.

    Usage:
        server = VsockServer(use_vsock=False)  # TCP mode for local dev
        server.register_handler("echo", handle_echo)
        await server.start()
    """

    def __init__(
        self,
        *,
        use_vsock: bool = False,
        port: int | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self._use_vsock = use_vsock
        self._port = port or (DEFAULT_VSOCK_PORT if use_vsock else DEFAULT_TCP_PORT)
        self._host = host
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, MessageFrame | None]]] = {}
        self._server: asyncio.AbstractServer | None = None
        self._shutdown_event = asyncio.Event()

    def register_handler(
        self,
        msg_type: str,
        handler: Callable[..., Coroutine[Any, Any, MessageFrame | None]],
    ) -> None:
        """Register a handler for a specific message type."""
        self._handlers[msg_type] = handler
        logger.info("handler_registered", extra={"msg_type": msg_type})

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        peer = writer.get_extra_info("peername", "unknown")
        logger.info("client_connected", extra={"peer": str(peer)})

        try:
            while not self._shutdown_event.is_set():
                try:
                    msg = await asyncio.wait_for(read_frame(reader), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.IncompleteReadError:
                    logger.info("client_disconnected", extra={"peer": str(peer)})
                    break

                logger.info(
                    "message_received",
                    extra={
                        "msg_type": msg.msg_type,
                        "request_id": msg.request_id,
                    },
                )

                handler = self._handlers.get(msg.msg_type)
                if handler is None:
                    response = MessageFrame(
                        msg_type="error",
                        payload={"error": f"Unknown message type: {msg.msg_type}"},
                        request_id=msg.request_id,
                    )
                else:
                    try:
                        import inspect
                        sig = inspect.signature(handler)
                        if "writer" in sig.parameters:
                            response = await handler(msg, writer=writer)
                        else:
                            response = await handler(msg)
                    except Exception as exc:
                        logger.error(
                            "handler_error",
                            extra={
                                "msg_type": msg.msg_type,
                                "error": str(exc),
                            },
                        )
                        response = MessageFrame(
                            msg_type="error",
                            payload={"error": f"Handler error: {exc}"},
                            request_id=msg.request_id,
                        )

                if response is not None:
                    await write_frame(writer, response)

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        """Start the server and listen for connections."""
        if self._use_vsock:
            await self._start_vsock()
        else:
            await self._start_tcp()

    async def _start_tcp(self) -> None:
        """Start a TCP server for local development."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )
        logger.info(
            "tcp_server_started",
            extra={"host": self._host, "port": self._port},
        )

        async with self._server:
            await self._server.serve_forever()

    async def _start_vsock(self) -> None:
        """Start a vsock server for production (inside enclave)."""
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.bind((VMADDR_CID_ANY, self._port))
        sock.setblocking(False)

        self._server = await asyncio.start_server(
            self._handle_connection,
            sock=sock,
        )
        logger.info("vsock_server_started", extra={"port": self._port})

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        self._shutdown_event.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("server_stopped")
