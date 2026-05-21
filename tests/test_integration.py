"""Integration tests — full end-to-end: host API → vsock → enclave → agent → response.

These tests spin up a real enclave service on TCP and test the full flow.
"""

from __future__ import annotations

import asyncio
import pytest

from enclave.main import EnclaveConfig, EnclaveService
from enclave.vsock.client import VsockClient
from enclave.vsock.protocol import MessageFrame


class TestEnclaveServiceIntegration:
    """Test the full enclave service end-to-end."""

    @pytest.fixture
    async def service_and_client(self):
        """Start an enclave service on a random TCP port and return a client."""
        config = EnclaveConfig()
        config.use_vsock = False
        config.tcp_port = 0  # Will be assigned by server
        config.tcp_host = "127.0.0.1"
        config.llm_provider = "mock"
        config.default_timeout = 10.0  # Short timeout for tests

        service = EnclaveService(config)
        # Override state_db to use in-memory database (avoid stale file data)
        from enclave.memory.state_db import TaskStateDB
        service.state_db = TaskStateDB()  # :memory:
        await service.state_db.initialize()

        # Start server in background — use a specific port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        actual_port = sock.getsockname()[1]
        sock.close()

        config.tcp_port = actual_port
        service.server._port = actual_port

        server_task = asyncio.create_task(service.server.start())

        # Wait for server to be ready
        await asyncio.sleep(0.3)

        client = VsockClient(
            use_vsock=False,
            host="127.0.0.1",
            port=actual_port,
        )

        yield service, client

        await service.server.stop()
        await service.state_db.close()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_echo(self, service_and_client) -> None:
        """Test echo round-trip through vsock."""
        _, client = service_and_client
        response = await client.send(MessageFrame(
            msg_type="echo",
            payload={"hello": "world"},
        ))
        assert response.msg_type == "echo_response"
        assert response.payload["hello"] == "world"

    @pytest.mark.asyncio
    async def test_status(self, service_and_client) -> None:
        """Test status endpoint."""
        _, client = service_and_client
        response = await client.send(MessageFrame(
            msg_type="status",
            payload={},
        ))
        assert response.msg_type == "status_response"
        assert response.payload["status"] == "running"
        assert "code_exec" in response.payload["tools"]
        assert response.payload["llm_provider"] == "mock"

    @pytest.mark.asyncio
    async def test_attestation(self, service_and_client) -> None:
        """Test attestation document retrieval."""
        _, client = service_and_client
        response = await client.send(MessageFrame(
            msg_type="attest",
            payload={"nonce": "test123"},
        ))
        assert response.msg_type == "attestation"
        assert "0" in response.payload["pcrs"]
        assert "verify_key" in response.payload
        assert "public_key" in response.payload

    @pytest.mark.asyncio
    async def test_full_task_execution(self, service_and_client) -> None:
        """Test a complete task execution through vsock."""
        service, client = service_and_client
        response = await client.send(
            MessageFrame(
                msg_type="task_request",
                payload={
                    "task_id": "integ_test_001",
                    "user_id": "test_user",
                    "description": "Say hello",
                    "max_steps": 5,
                    "budget_usd": 1.0,
                },
            ),
            timeout=30.0,
        )
        assert response.msg_type == "task_result"
        assert response.payload["task_id"] == "integ_test_001"
        assert response.payload["success"] is True
        assert "attestation" in response.payload
        assert response.payload["attestation"]["task_hash"] != ""

        # Verify task was recorded in state DB
        task = await service.state_db.get_task("integ_test_001")
        assert task is not None
        assert task["status"] == "completed"

    @pytest.mark.asyncio
    async def test_empty_description_rejected(self, service_and_client) -> None:
        """Test that empty task descriptions are rejected."""
        _, client = service_and_client
        response = await client.send(MessageFrame(
            msg_type="task_request",
            payload={
                "task_id": "empty_test",
                "description": "",
            },
        ))
        assert response.msg_type == "task_error"
        assert "empty" in response.payload.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_unknown_message_type(self, service_and_client) -> None:
        """Test unknown message types return errors."""
        _, client = service_and_client
        response = await client.send(MessageFrame(
            msg_type="nonexistent_handler",
            payload={},
        ))
        assert response.msg_type == "error"
        assert "Unknown" in response.payload.get("error", "")

    @pytest.mark.asyncio
    async def test_attestation_receipt_verifiable(self, service_and_client) -> None:
        """Test that attestation receipts can be verified by the host."""
        from host.attestation.verifier import AttestationVerifier

        service, client = service_and_client

        # Get attestation document to know expected PCRs
        attest_resp = await client.send(MessageFrame(
            msg_type="attest",
            payload={},
        ))
        expected_pcrs = {
            int(k): v for k, v in attest_resp.payload["pcrs"].items()
        }

        # Run a task
        task_resp = await client.send(
            MessageFrame(
                msg_type="task_request",
                payload={
                    "task_id": "verify_test",
                    "description": "Test verification",
                },
            ),
            timeout=30.0,
        )
        receipt = task_resp.payload["attestation"]

        # Verify with host verifier
        verifier = AttestationVerifier(expected_pcrs=expected_pcrs)
        assert verifier.verify_task_integrity(
            receipt["task_hash"],
            receipt,
        ) is True

    @pytest.mark.asyncio
    async def test_multiple_sequential_tasks(self, service_and_client) -> None:
        """Test running multiple tasks sequentially on the same connection."""
        service, client = service_and_client

        for i in range(3):
            response = await client.send(
                MessageFrame(
                    msg_type="task_request",
                    payload={
                        "task_id": f"seq_task_{i}",
                        "description": f"Sequential task {i}",
                    },
                ),
                timeout=15.0,
            )
            assert response.payload["success"] is True

        # Verify all tasks recorded
        tasks = await service.state_db.list_tasks()
        task_ids = {t["task_id"] for t in tasks}
        for i in range(3):
            assert f"seq_task_{i}" in task_ids


class TestEnclaveConfig:
    """Test configuration loading."""

    def test_default_config(self) -> None:
        config = EnclaveConfig()
        assert config.use_vsock is False
        assert config.llm_provider == "mock"
        assert config.default_max_steps == 50

    def test_domain_allowlist_parsing(self) -> None:
        import os
        original = os.environ.get("ENCLAVE_DOMAIN_ALLOWLIST")
        try:
            os.environ["ENCLAVE_DOMAIN_ALLOWLIST"] = "api.github.com, pypi.org, example.com"
            config = EnclaveConfig()
            assert config.domain_allowlist == ["api.github.com", "pypi.org", "example.com"]
        finally:
            if original:
                os.environ["ENCLAVE_DOMAIN_ALLOWLIST"] = original
            else:
                os.environ.pop("ENCLAVE_DOMAIN_ALLOWLIST", None)

    def test_empty_allowlist(self) -> None:
        config = EnclaveConfig()
        assert config.domain_allowlist is None
