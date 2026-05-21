"""Tests for cryptographic primitives, attestation, sealing, and memory encryption."""

from __future__ import annotations

import os
import pytest

from enclave.crypto.keys import EnclaveKeyManager, seal_data, unseal_data
from enclave.crypto.attestation import MockAttestationProvider, AttestationDocument
from enclave.crypto.sealing import SealedSecretStore, SealingError


class TestEnclaveKeyManager:
    """Test key generation and derivation."""

    def test_key_generation(self) -> None:
        km = EnclaveKeyManager()
        keys = km.keys
        assert len(keys.master_secret) == 32
        assert keys.signing_key is not None
        assert keys.verify_key is not None

    def test_sign_and_verify(self) -> None:
        km = EnclaveKeyManager()
        data = b"Hello, enclave!"
        signed = km.sign(data)
        verified = km.verify(signed)
        assert verified == data

    def test_verify_tampered_fails(self) -> None:
        km = EnclaveKeyManager()
        data = b"Original data"
        signed = bytearray(km.sign(data))
        signed[-1] ^= 0xFF  # Tamper with the last byte
        with pytest.raises(Exception):
            km.verify(bytes(signed))

    def test_seal_unseal_roundtrip(self) -> None:
        key = os.urandom(32)
        plaintext = b"Secret data that must be protected"
        ciphertext = seal_data(plaintext, key)
        assert ciphertext != plaintext
        decrypted = unseal_data(ciphertext, key)
        assert decrypted == plaintext

    def test_unseal_wrong_key_fails(self) -> None:
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        plaintext = b"Secret"
        ciphertext = seal_data(plaintext, key1)
        with pytest.raises(Exception):
            unseal_data(ciphertext, key2)

    def test_key_derivation_deterministic(self) -> None:
        master = os.urandom(32)
        key1 = EnclaveKeyManager.derive_key(master, "test_context", 1)
        key2 = EnclaveKeyManager.derive_key(master, "test_context", 1)
        assert key1 == key2

    def test_key_derivation_different_contexts(self) -> None:
        master = os.urandom(32)
        key1 = EnclaveKeyManager.derive_key(master, "context_a", 1)
        key2 = EnclaveKeyManager.derive_key(master, "context_b", 1)
        assert key1 != key2

    def test_key_derivation_different_subkeys(self) -> None:
        master = os.urandom(32)
        key1 = EnclaveKeyManager.derive_key(master, "same_context", 1)
        key2 = EnclaveKeyManager.derive_key(master, "same_context", 2)
        assert key1 != key2

    def test_sealed_box_encrypt_decrypt(self) -> None:
        km = EnclaveKeyManager()
        plaintext = b"Encrypted for the enclave"
        ciphertext = km.encrypt_sealed(plaintext, km.keys.public_key)
        decrypted = km.decrypt_sealed(ciphertext)
        assert decrypted == plaintext


class TestMockAttestationProvider:
    """Test mock attestation document generation."""

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        provider = MockAttestationProvider()
        doc = await provider.get_document()
        assert isinstance(doc, AttestationDocument)
        assert 0 in doc.pcrs
        assert 1 in doc.pcrs
        assert 2 in doc.pcrs
        assert len(doc.pcrs[0]) == 48  # SHA-384 output

    @pytest.mark.asyncio
    async def test_pcr_deterministic(self) -> None:
        provider = MockAttestationProvider(image_hash="test-v1")
        pcrs1 = await provider.get_pcr_values()
        pcrs2 = await provider.get_pcr_values()
        assert pcrs1 == pcrs2

    @pytest.mark.asyncio
    async def test_different_image_different_pcrs(self) -> None:
        p1 = MockAttestationProvider(image_hash="image-v1")
        p2 = MockAttestationProvider(image_hash="image-v2")
        pcrs1 = await p1.get_pcr_values()
        pcrs2 = await p2.get_pcr_values()
        assert pcrs1[0] != pcrs2[0]

    @pytest.mark.asyncio
    async def test_tampered_changes_pcr0(self) -> None:
        provider = MockAttestationProvider(image_hash="test")
        normal = await provider.get_pcr_values()
        provider.set_tampered(True)
        tampered = await provider.get_pcr_values()
        assert normal[0] != tampered[0]
        # PCR[1] and PCR[2] should be unchanged
        assert normal[1] == tampered[1]

    @pytest.mark.asyncio
    async def test_user_data_included(self) -> None:
        provider = MockAttestationProvider()
        user_data = b"custom-nonce-12345"
        doc = await provider.get_document(user_data=user_data)
        assert doc.user_data == user_data


class TestSealedSecretStore:
    """Test secret sealing with PCR-derived keys."""

    @pytest.mark.asyncio
    async def test_seal_unseal_matching_pcrs(self) -> None:
        provider = MockAttestationProvider()
        pcrs = await provider.get_pcr_values()

        store = SealedSecretStore()
        secret = b"my-master-secret-key-32-bytes!!"
        sealed = store.seal(secret, pcrs)
        unsealed = store.unseal(sealed, pcrs)
        assert unsealed == secret

    @pytest.mark.asyncio
    async def test_unseal_mismatched_pcrs_fails(self) -> None:
        """Changing the enclave image (PCRs) makes secrets unreadable."""
        provider = MockAttestationProvider(image_hash="v1")
        pcrs_v1 = await provider.get_pcr_values()

        store = SealedSecretStore()
        secret = b"secret-sealed-to-v1-image!!!!!"
        sealed = store.seal(secret, pcrs_v1)

        # Simulate enclave image update (different PCRs)
        provider_v2 = MockAttestationProvider(image_hash="v2")
        pcrs_v2 = await provider_v2.get_pcr_values()

        with pytest.raises(SealingError, match="PCR values may have changed"):
            store.unseal(sealed, pcrs_v2)

    @pytest.mark.asyncio
    async def test_tampered_enclave_fails(self) -> None:
        """Tampering with the enclave changes PCRs, preventing secret access."""
        provider = MockAttestationProvider()
        pcrs_clean = await provider.get_pcr_values()

        store = SealedSecretStore()
        secret = b"important-secret-data-here!!!!"
        sealed = store.seal(secret, pcrs_clean)

        # Simulate tampered enclave
        provider.set_tampered(True)
        pcrs_tampered = await provider.get_pcr_values()

        with pytest.raises(SealingError):
            store.unseal(sealed, pcrs_tampered)

    @pytest.mark.asyncio
    async def test_seal_to_file_and_unseal(self, tmp_path) -> None:
        provider = MockAttestationProvider()
        pcrs = await provider.get_pcr_values()

        store = SealedSecretStore(storage_path=tmp_path)
        secret = b"file-sealed-secret-32-bytes!!!!"
        file_path = store.seal_to_file(secret, pcrs, label="test_key")
        assert file_path.exists()

        unsealed = store.unseal_from_file(pcrs, label="test_key")
        assert unsealed == secret


class TestMemoryEncryption:
    """Test the encrypted memory manager."""

    def test_store_and_search(self) -> None:
        from enclave.memory.manager import MemoryManager

        master = os.urandom(32)
        mgr = MemoryManager(master)

        entry_id = mgr.store("Python is a great language", collection="facts")
        assert entry_id is not None

        results = mgr.search("Python language", collection="facts")
        assert len(results) > 0

    def test_encrypted_entries_verifiable(self) -> None:
        from enclave.memory.manager import MemoryManager

        master = os.urandom(32)
        mgr = MemoryManager(master)

        mgr.store("Secret: API key is abc123", collection="secrets")
        results = mgr.search("API key", collection="secrets")
        assert len(results) > 0

        # Verify the encrypted entry can be decrypted
        decrypted = mgr.verify_encrypted_entry(results[0], collection="secrets")
        assert decrypted is not None
        assert "abc123" in decrypted

    def test_wrong_key_cannot_decrypt(self) -> None:
        from enclave.memory.manager import MemoryManager

        master1 = os.urandom(32)
        master2 = os.urandom(32)
        mgr1 = MemoryManager(master1)
        mgr2 = MemoryManager(master2)

        mgr1.store("Secret data", collection="test")
        results = mgr1.search("Secret", collection="test")
        assert len(results) > 0

        # Different master key cannot decrypt
        decrypted = mgr2.verify_encrypted_entry(results[0], collection="test")
        assert decrypted is None

    def test_collection_isolation(self) -> None:
        from enclave.memory.manager import MemoryManager

        master = os.urandom(32)
        mgr = MemoryManager(master)

        mgr.store("Data in collection A", collection="col_a")
        mgr.store("Data in collection B", collection="col_b")

        assert mgr.collection_count("col_a") == 1
        assert mgr.collection_count("col_b") == 1
        assert "col_a" in mgr.list_collections()
        assert "col_b" in mgr.list_collections()


class TestVsockProtocol:
    """Test vsock message framing."""

    def test_encode_decode_roundtrip(self) -> None:
        from enclave.vsock.protocol import MessageFrame, encode_frame, decode_frame

        msg = MessageFrame(
            msg_type="echo",
            payload={"data": "hello", "number": 42},
            request_id="abc123",
        )
        encoded = encode_frame(msg)
        decoded = decode_frame(encoded)

        assert decoded.msg_type == "echo"
        assert decoded.payload["data"] == "hello"
        assert decoded.payload["number"] == 42
        assert decoded.request_id == "abc123"

    def test_empty_payload(self) -> None:
        from enclave.vsock.protocol import MessageFrame, encode_frame, decode_frame

        msg = MessageFrame(msg_type="ping", payload={})
        encoded = encode_frame(msg)
        decoded = decode_frame(encoded)
        assert decoded.msg_type == "ping"

    def test_oversized_message_rejected(self) -> None:
        from enclave.vsock.protocol import MessageFrame, FramingError, encode_frame

        msg = MessageFrame(
            msg_type="huge",
            payload={"data": "x" * (17 * 1024 * 1024)},  # > 16MB
        )
        with pytest.raises(FramingError, match="exceeds maximum"):
            encode_frame(msg)

    def test_truncated_data_rejected(self) -> None:
        from enclave.vsock.protocol import FramingError, decode_frame

        with pytest.raises(FramingError):
            decode_frame(b"\x00")  # Too short for header

    def test_invalid_json_rejected(self) -> None:
        import struct

        from enclave.vsock.protocol import FramingError, decode_frame

        bad_payload = b"not json at all"
        header = struct.pack("<I", len(bad_payload))
        with pytest.raises(FramingError, match="Invalid JSON"):
            decode_frame(header + bad_payload)


class TestTaskStateDB:
    """Test the task state database."""

    @pytest.mark.asyncio
    async def test_create_and_get_task(self) -> None:
        from enclave.memory.state_db import TaskStateDB
        from enclave.agent.models import TaskRequest

        db = TaskStateDB()
        await db.initialize()

        task = TaskRequest(
            description="Test task",
            task_id="test123",
            user_id="user1",
        )
        await db.create_task(task)

        retrieved = await db.get_task("test123")
        assert retrieved is not None
        assert retrieved["task_id"] == "test123"
        assert retrieved["status"] == "running"

    @pytest.mark.asyncio
    async def test_complete_task(self) -> None:
        from enclave.memory.state_db import TaskStateDB
        from enclave.agent.models import TaskRequest, TaskResult

        db = TaskStateDB()
        await db.initialize()

        task = TaskRequest(task_id="t1", description="Test", user_id="u1")
        await db.create_task(task)

        result = TaskResult(
            task_id="t1",
            success=True,
            summary="Done",
            total_cost_usd=0.05,
            attestation_hash="abc",
        )
        await db.complete_task(result)

        retrieved = await db.get_task("t1")
        assert retrieved["status"] == "completed"
        assert retrieved["success"] == 1

    @pytest.mark.asyncio
    async def test_list_tasks(self) -> None:
        from enclave.memory.state_db import TaskStateDB
        from enclave.agent.models import TaskRequest

        db = TaskStateDB()
        await db.initialize()

        for i in range(5):
            task = TaskRequest(task_id=f"t{i}", description=f"Task {i}", user_id="u1")
            await db.create_task(task)

        tasks = await db.list_tasks(user_id="u1")
        assert len(tasks) == 5

    @pytest.mark.asyncio
    async def test_nonexistent_task(self) -> None:
        from enclave.memory.state_db import TaskStateDB

        db = TaskStateDB()
        await db.initialize()

        result = await db.get_task("nonexistent")
        assert result is None
