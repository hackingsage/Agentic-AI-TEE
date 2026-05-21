"""Vsock client — async client for host-to-enclave communication.

Supports pluggable transport: AF_VSOCK (production) or TCP (local dev).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from typing import Any

from enclave.vsock.protocol import MessageFrame, read_frame, write_frame

logger = logging.getLogger(__name__)

DEFAULT_VSOCK_PORT = 5000
DEFAULT_TCP_PORT = 8765
DEFAULT_TIMEOUT = 30.0


class VsockClient:
    """Async client for communicating with the enclave server.

    In production, connects via AF_VSOCK. For local development,
    connects via TCP to localhost.

    Usage:
        client = VsockClient(use_vsock=False)  # TCP mode
        response = await client.send(MessageFrame(msg_type="echo", payload={"data": "hello"}))
    """

    def __init__(
        self,
        *,
        use_vsock: bool = False,
        enclave_cid: int = 0,
        port: int | None = None,
        host: str = "127.0.0.1",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._use_vsock = use_vsock
        self._enclave_cid = enclave_cid
        self._port = port or (DEFAULT_VSOCK_PORT if use_vsock else DEFAULT_TCP_PORT)
        self._host = host
        self._timeout = timeout

    async def send(
        self,
        msg: MessageFrame,
        *,
        timeout: float | None = None,
    ) -> MessageFrame:
        """Send a message and wait for the response.

        Args:
            msg: The message to send.
            timeout: Response timeout in seconds.

        Returns:
            The response MessageFrame.

        Raises:
            ConnectionError: If connection fails.
            asyncio.TimeoutError: If response times out.
        """
        effective_timeout = timeout or self._timeout

        # Ensure request_id is set for correlation
        if not msg.request_id:
            msg.request_id = uuid.uuid4().hex[:12]

        reader, writer = await self._connect()

        try:
            await write_frame(writer, msg)

            response = await asyncio.wait_for(
                read_frame(reader),
                timeout=effective_timeout,
            )

            return response

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def send_no_response(self, msg: MessageFrame) -> None:
        """Send a message without waiting for a response (fire-and-forget)."""
        if not msg.request_id:
            msg.request_id = uuid.uuid4().hex[:12]

        reader, writer = await self._connect()

        try:
            await write_frame(writer, msg)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Establish a connection to the enclave server."""
        if self._use_vsock:
            return await self._connect_vsock()
        else:
            return await self._connect_tcp()

    async def _connect_tcp(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect via TCP for local development."""
        return await asyncio.open_connection(self._host, self._port)

    async def _connect_vsock(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect via AF_VSOCK for production."""
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.setblocking(False)

        loop = asyncio.get_event_loop()
        await loop.sock_connect(sock, (self._enclave_cid, self._port))

        reader, writer = await asyncio.open_connection(sock=sock)
        return reader, writer
