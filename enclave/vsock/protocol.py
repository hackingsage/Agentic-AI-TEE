"""Vsock message framing protocol.

Implements length-prefix framing for vsock communication:
    [4-byte little-endian length][JSON payload]

Validates message sizes to prevent DoS attacks.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum message size: 16MB
MAX_MESSAGE_SIZE = 16 * 1024 * 1024
HEADER_SIZE = 4  # 4 bytes for uint32 LE length


@dataclass
class MessageFrame:
    """A framed message for vsock transport."""

    msg_type: str  # "task_request", "task_result", "step_event", "attest", "echo", etc.
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "msg_type": self.msg_type,
            "payload": self.payload,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageFrame:
        return cls(
            msg_type=data.get("msg_type", ""),
            payload=data.get("payload", {}),
            request_id=data.get("request_id", ""),
        )


class FramingError(Exception):
    """Raised when message framing is invalid."""


def encode_frame(msg: MessageFrame) -> bytes:
    """Encode a MessageFrame into a length-prefixed byte string.

    Format: [4-byte LE uint32 length][JSON payload bytes]

    Raises:
        FramingError: If the encoded message exceeds MAX_MESSAGE_SIZE.
    """
    payload_bytes = json.dumps(msg.to_dict(), separators=(",", ":")).encode("utf-8")

    if len(payload_bytes) > MAX_MESSAGE_SIZE:
        raise FramingError(
            f"Message size {len(payload_bytes)} exceeds maximum {MAX_MESSAGE_SIZE}"
        )

    header = struct.pack("<I", len(payload_bytes))
    return header + payload_bytes


def decode_frame(data: bytes) -> MessageFrame:
    """Decode a length-prefixed byte string into a MessageFrame.

    Args:
        data: Raw bytes including the 4-byte length header.

    Returns:
        Parsed MessageFrame.

    Raises:
        FramingError: If the data is malformed or exceeds size limits.
    """
    if len(data) < HEADER_SIZE:
        raise FramingError(f"Data too short for header: {len(data)} < {HEADER_SIZE}")

    payload_length = struct.unpack("<I", data[:HEADER_SIZE])[0]

    if payload_length > MAX_MESSAGE_SIZE:
        raise FramingError(
            f"Declared payload size {payload_length} exceeds maximum {MAX_MESSAGE_SIZE}"
        )

    if len(data) < HEADER_SIZE + payload_length:
        raise FramingError(
            f"Incomplete message: expected {HEADER_SIZE + payload_length} bytes, "
            f"got {len(data)}"
        )

    payload_bytes = data[HEADER_SIZE : HEADER_SIZE + payload_length]

    try:
        payload_dict = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FramingError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload_dict, dict):
        raise FramingError(f"Payload must be a JSON object, got {type(payload_dict).__name__}")

    return MessageFrame.from_dict(payload_dict)


async def read_frame(reader: Any) -> MessageFrame:
    """Read a single framed message from an async stream reader.

    Args:
        reader: asyncio.StreamReader or compatible object with readexactly().

    Returns:
        Parsed MessageFrame.

    Raises:
        FramingError: If the message is malformed.
        asyncio.IncompleteReadError: If the connection closes mid-message.
    """
    # Read the 4-byte length header
    header = await reader.readexactly(HEADER_SIZE)
    payload_length = struct.unpack("<I", header)[0]

    # Validate before allocating buffer (DoS prevention)
    if payload_length > MAX_MESSAGE_SIZE:
        raise FramingError(
            f"Declared payload size {payload_length} exceeds maximum {MAX_MESSAGE_SIZE}"
        )

    if payload_length == 0:
        raise FramingError("Empty payload")

    # Read the payload
    payload_bytes = await reader.readexactly(payload_length)

    try:
        payload_dict = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FramingError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload_dict, dict):
        raise FramingError(f"Payload must be a JSON object, got {type(payload_dict).__name__}")

    return MessageFrame.from_dict(payload_dict)


async def write_frame(writer: Any, msg: MessageFrame) -> None:
    """Write a single framed message to an async stream writer.

    Args:
        writer: asyncio.StreamWriter or compatible object with write() and drain().
        msg: The MessageFrame to send.
    """
    frame_bytes = encode_frame(msg)
    writer.write(frame_bytes)
    await writer.drain()
