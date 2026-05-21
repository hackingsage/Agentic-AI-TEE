"""Sealed secret store — encrypts secrets with PCR-derived keys.

If the enclave image changes (PCRs change), sealed secrets become
unreadable, preventing access from tampered enclaves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from enclave.crypto.keys import seal_data, unseal_data, EnclaveKeyManager

logger = logging.getLogger(__name__)

# Size constants
SALT_SIZE = 32


class SealingError(Exception):
    """Raised when sealing or unsealing fails."""


class SealedSecretStore:
    """Encrypts secrets with PCR-derived keys.

    The encryption key is derived from PCR measurements, so if the
    enclave image is modified (changing PCRs), the sealed data
    becomes unreadable.

    This provides a "key sealing" mechanism: secrets are bound to
    a specific enclave identity.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = storage_path or Path("/tmp/enclave_sealed")
        self._storage_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pcr_to_key(pcr_values: dict[int, bytes], salt: bytes | None = None) -> bytes:
        """Derive a 32-byte encryption key from PCR values.

        Concatenates all PCR values in order and derives a key using
        HKDF-SHA256 with an optional salt.
        """
        # Concatenate PCR values in deterministic order
        pcr_material = b""
        for idx in sorted(pcr_values.keys()):
            pcr_material += pcr_values[idx]

        # Use HKDF to derive the key
        return EnclaveKeyManager.derive_key(
            master_secret=hashlib.sha256(pcr_material).digest(),
            context="sealed_secret_store",
            subkey_id=1,
        )

    def seal(
        self,
        secret: bytes,
        pcr_values: dict[int, bytes],
        label: str = "master_secret",
    ) -> bytes:
        """Seal a secret with PCR-derived encryption.

        Args:
            secret: The secret to seal.
            pcr_values: Current PCR measurements.
            label: Label for the sealed data (for storage).

        Returns:
            The encrypted (sealed) bytes.
        """
        key = self._pcr_to_key(pcr_values)
        sealed = seal_data(secret, key)

        logger.info(
            "secret_sealed",
            extra={"label": label, "sealed_size": len(sealed)},
        )

        return sealed

    def unseal(
        self,
        sealed_data: bytes,
        pcr_values: dict[int, bytes],
        label: str = "master_secret",
    ) -> bytes:
        """Unseal a secret using PCR-derived encryption.

        Args:
            sealed_data: The encrypted (sealed) bytes.
            pcr_values: Current PCR measurements (must match sealing PCRs).
            label: Label for logging.

        Returns:
            The original secret.

        Raises:
            SealingError: If decryption fails (PCRs changed or data tampered).
        """
        key = self._pcr_to_key(pcr_values)

        try:
            secret = unseal_data(sealed_data, key)
            logger.info(
                "secret_unsealed",
                extra={"label": label, "secret_size": len(secret)},
            )
            return secret
        except Exception as exc:
            raise SealingError(
                f"Failed to unseal '{label}': PCR values may have changed "
                f"or data is tampered. Error: {exc}"
            ) from exc

    def seal_to_file(
        self,
        secret: bytes,
        pcr_values: dict[int, bytes],
        label: str = "master_secret",
    ) -> Path:
        """Seal a secret and write it to a file."""
        sealed = self.seal(secret, pcr_values, label)
        file_path = self._storage_path / f"{label}.sealed"
        file_path.write_bytes(sealed)
        return file_path

    def unseal_from_file(
        self,
        pcr_values: dict[int, bytes],
        label: str = "master_secret",
    ) -> bytes:
        """Read and unseal a secret from a file."""
        file_path = self._storage_path / f"{label}.sealed"
        if not file_path.exists():
            raise SealingError(f"Sealed file not found: {file_path}")

        sealed = file_path.read_bytes()
        return self.unseal(sealed, pcr_values, label)
