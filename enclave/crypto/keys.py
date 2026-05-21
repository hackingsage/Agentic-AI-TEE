"""Cryptographic key management for the enclave.

Provides key generation, derivation, and symmetric encryption using
libsodium (via PyNaCl) and the cryptography library for HKDF.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import nacl.secret
import nacl.signing
import nacl.utils
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.public import PrivateKey, PublicKey, SealedBox

logger = logging.getLogger(__name__)

# Key sizes
MASTER_KEY_SIZE = 32  # 256-bit master key
DERIVED_KEY_SIZE = 32  # 256-bit derived keys
NONCE_SIZE = 24  # NaCl nonce size


@dataclass
class EnclaveKeys:
    """Container for all enclave cryptographic keys."""

    signing_key: nacl.signing.SigningKey  # Ed25519
    verify_key: nacl.signing.VerifyKey  # Ed25519 public
    encryption_key: PrivateKey  # Curve25519
    public_key: PublicKey  # Curve25519 public
    master_secret: bytes  # 256-bit master secret


class EnclaveKeyManager:
    """Manages enclave cryptographic keys.

    Generates Ed25519 signing keys and Curve25519 encryption keys.
    Derives per-task and per-collection keys from a master secret.
    """

    def __init__(self, master_secret: bytes | None = None) -> None:
        """Initialize key manager.

        Args:
            master_secret: 32-byte master secret. Generated randomly if not provided.
                In production, this is unsealed from the SealedSecretStore.
        """
        self._master_secret = master_secret or os.urandom(MASTER_KEY_SIZE)

        # Generate signing keypair (Ed25519) from master secret
        # Derive a separate seed for signing to avoid key reuse across algorithms
        signing_seed = self.derive_key(self._master_secret, "ed25519_signing", 1)
        self._signing_key = nacl.signing.SigningKey(signing_seed)
        self._verify_key = self._signing_key.verify_key

        # Generate encryption keypair (Curve25519) from a different derivation
        encryption_seed = self.derive_key(self._master_secret, "curve25519_encryption", 1)
        self._encryption_key = PrivateKey(encryption_seed)
        self._public_key = self._encryption_key.public_key

        logger.info(
            "keys_initialized",
            extra={"verify_key": self._verify_key.encode().hex()[:16] + "..."},
        )

    @property
    def keys(self) -> EnclaveKeys:
        """Return all enclave keys."""
        return EnclaveKeys(
            signing_key=self._signing_key,
            verify_key=self._verify_key,
            encryption_key=self._encryption_key,
            public_key=self._public_key,
            master_secret=self._master_secret,
        )

    def sign(self, data: bytes) -> bytes:
        """Sign data with the Ed25519 signing key. Returns signature + data."""
        signed = self._signing_key.sign(data)
        return bytes(signed)

    def sign_detached(self, data: bytes) -> bytes:
        """Sign data and return only the signature (no message)."""
        signed = self._signing_key.sign(data)
        return bytes(signed.signature)

    def verify(self, signed_data: bytes) -> bytes:
        """Verify signed data. Returns the original data. Raises on failure."""
        return bytes(self._verify_key.verify(signed_data))

    def encrypt_sealed(self, plaintext: bytes, recipient_public_key: PublicKey) -> bytes:
        """Encrypt data for a recipient using SealedBox (anonymous sender)."""
        box = SealedBox(recipient_public_key)
        return bytes(box.encrypt(plaintext))

    def decrypt_sealed(self, ciphertext: bytes) -> bytes:
        """Decrypt data sent to this enclave via SealedBox."""
        box = SealedBox(self._encryption_key)
        return bytes(box.decrypt(ciphertext))

    @staticmethod
    def derive_key(
        master_secret: bytes,
        context: str,
        subkey_id: int = 1,
    ) -> bytes:
        """Derive a subkey from the master secret using HKDF-SHA256.

        Args:
            master_secret: The master secret (32 bytes).
            context: Context string (e.g., "task:abc123", "collection:memories").
            subkey_id: Numeric subkey identifier for key separation.

        Returns:
            32-byte derived key.
        """
        info = f"{context}:{subkey_id}".encode("utf-8")
        hkdf = HKDF(
            algorithm=SHA256(),
            length=DERIVED_KEY_SIZE,
            salt=None,
            info=info,
        )
        return hkdf.derive(master_secret)


def seal_data(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt data using NaCl SecretBox (XSalsa20-Poly1305).

    Args:
        plaintext: Data to encrypt.
        key: 32-byte symmetric key.

    Returns:
        Nonce + ciphertext (nonce is prepended automatically by SecretBox).
    """
    box = nacl.secret.SecretBox(key)
    return bytes(box.encrypt(plaintext))


def unseal_data(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt data using NaCl SecretBox.

    Args:
        ciphertext: Encrypted data (nonce + ciphertext).
        key: 32-byte symmetric key.

    Returns:
        Decrypted plaintext.

    Raises:
        nacl.exceptions.CryptoError: If decryption fails (wrong key or tampered data).
    """
    box = nacl.secret.SecretBox(key)
    return bytes(box.decrypt(ciphertext))
