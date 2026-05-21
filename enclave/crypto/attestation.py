"""Attestation framework — pluggable attestation providers.

Supports AWS Nitro NSM (production) and mock attestation (local dev).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AttestationDocument:
    """Parsed attestation document."""

    pcrs: dict[int, bytes]  # PCR index → measurement hash
    certificate: bytes  # DER-encoded certificate
    timestamp: float
    user_data: bytes | None = None
    nonce: bytes | None = None
    public_key: bytes | None = None
    raw_document: bytes = b""  # Original COSE_Sign1 document


@dataclass
class AttestationReceipt:
    """Attestation receipt for a completed task."""

    task_id: str
    task_hash: str  # SHA3-256 of task prompt + all outputs
    attestation_document: AttestationDocument
    enclave_signature: bytes  # Ed25519 signature of (task_hash + attestation_doc)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "pcrs": {
                str(k): v.hex() for k, v in self.attestation_document.pcrs.items()
            },
            "timestamp": self.timestamp,
            "signature": self.enclave_signature.hex(),
        }


class AttestationProvider(ABC):
    """Abstract attestation provider interface."""

    @abstractmethod
    async def get_document(
        self,
        user_data: bytes | None = None,
        nonce: bytes | None = None,
        public_key: bytes | None = None,
    ) -> AttestationDocument:
        """Get an attestation document from the TEE.

        Args:
            user_data: Optional user data to include in the document.
            nonce: Optional nonce for freshness.
            public_key: Optional public key to bind to the attestation.

        Returns:
            AttestationDocument with PCR values and certificate.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_pcr_values(self) -> dict[int, bytes]:
        """Get current PCR measurements."""
        raise NotImplementedError


class NitroAttestationProvider(AttestationProvider):
    """AWS Nitro Enclave attestation provider.

    Uses the NSM (Nitro Secure Module) library to get real attestation
    documents. Only works inside a Nitro Enclave.
    """

    async def get_document(
        self,
        user_data: bytes | None = None,
        nonce: bytes | None = None,
        public_key: bytes | None = None,
    ) -> AttestationDocument:
        try:
            # NSM library is only available inside Nitro Enclaves
            import nsm  # type: ignore[import-not-found]

            fd = nsm.nsm_lib_init()

            request: dict[str, Any] = {}
            if user_data:
                request["user_data"] = user_data
            if nonce:
                request["nonce"] = nonce
            if public_key:
                request["public_key"] = public_key

            response = nsm.nsm_get_attestation_doc(fd, **request)
            nsm.nsm_lib_exit(fd)

            # Parse COSE_Sign1 document
            import cbor2

            raw_doc = response
            cose_obj = cbor2.loads(raw_doc)

            # COSE_Sign1 = [protected, unprotected, payload, signature]
            payload = cbor2.loads(cose_obj[2]) if isinstance(cose_obj, list) else {}

            pcrs = {}
            if "pcrs" in payload:
                for idx, value in payload["pcrs"].items():
                    pcrs[int(idx)] = bytes(value)

            return AttestationDocument(
                pcrs=pcrs,
                certificate=payload.get("certificate", b""),
                timestamp=time.time(),
                user_data=user_data,
                nonce=nonce,
                public_key=public_key,
                raw_document=raw_doc,
            )

        except ImportError:
            raise RuntimeError(
                "NSM library not available. "
                "This provider only works inside a Nitro Enclave."
            )

    async def get_pcr_values(self) -> dict[int, bytes]:
        doc = await self.get_document()
        return doc.pcrs


class MockAttestationProvider(AttestationProvider):
    """Mock attestation provider for local development and testing.

    Generates deterministic PCR values from a configurable image hash.
    """

    def __init__(
        self,
        image_hash: str = "mock-enclave-image-v0.1.0",
        tampered: bool = False,
    ) -> None:
        self._image_hash = image_hash
        self._tampered = tampered

    def _generate_pcr(self, index: int) -> bytes:
        """Generate a deterministic PCR value for the given index."""
        seed = f"{self._image_hash}:pcr{index}"
        if self._tampered and index == 0:
            seed = f"TAMPERED:{seed}"
        return hashlib.sha384(seed.encode()).digest()

    async def get_document(
        self,
        user_data: bytes | None = None,
        nonce: bytes | None = None,
        public_key: bytes | None = None,
    ) -> AttestationDocument:
        pcrs = await self.get_pcr_values()

        # Generate a mock certificate
        cert_data = json.dumps({
            "type": "mock-attestation",
            "image_hash": self._image_hash,
            "timestamp": time.time(),
        }).encode()

        # Generate mock raw document (not a real COSE_Sign1)
        raw = json.dumps({
            "pcrs": {str(k): v.hex() for k, v in pcrs.items()},
            "certificate": cert_data.hex(),
            "user_data": user_data.hex() if user_data else None,
        }).encode()

        return AttestationDocument(
            pcrs=pcrs,
            certificate=cert_data,
            timestamp=time.time(),
            user_data=user_data,
            nonce=nonce,
            public_key=public_key,
            raw_document=raw,
        )

    async def get_pcr_values(self) -> dict[int, bytes]:
        return {
            0: self._generate_pcr(0),  # Image hash
            1: self._generate_pcr(1),  # Kernel hash
            2: self._generate_pcr(2),  # Application hash
        }

    def set_tampered(self, tampered: bool) -> None:
        """Toggle tampered mode for testing."""
        self._tampered = tampered
