"""Tests for the host API and attestation verifier."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from host.api.main import create_app
from host.attestation.verifier import AttestationVerifier, VerificationResult


class TestHostAPI:
    """Test the FastAPI host gateway."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health_check(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    def test_submit_task(self, client) -> None:
        response = client.post("/tasks", json={
            "description": "Write a hello world program",
            "budget_usd": 1.0,
            "max_steps": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    def test_submit_task_minimal(self, client) -> None:
        response = client.post("/tasks", json={
            "description": "Test task",
        })
        assert response.status_code == 200

    def test_submit_task_empty_description(self, client) -> None:
        response = client.post("/tasks", json={
            "description": "",
        })
        assert response.status_code == 422  # Validation error

    def test_get_task(self, client) -> None:
        # Submit first
        submit_resp = client.post("/tasks", json={"description": "Get test"})
        task_id = submit_resp.json()["task_id"]

        # Get
        response = client.get(f"/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "queued"

    def test_get_nonexistent_task(self, client) -> None:
        response = client.get("/tasks/nonexistent_id")
        assert response.status_code == 404

    def test_list_tasks(self, client) -> None:
        # Submit a few tasks
        for i in range(3):
            client.post("/tasks", json={"description": f"Task {i}"})

        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data
        assert data["total"] >= 3

    def test_attestation_not_completed(self, client) -> None:
        submit_resp = client.post("/tasks", json={"description": "Attest test"})
        task_id = submit_resp.json()["task_id"]

        response = client.get(f"/tasks/{task_id}/attestation")
        assert response.status_code == 400  # Not completed yet

    def test_enclave_attest(self, client) -> None:
        response = client.get("/enclave/attest")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAttestationVerifier:
    """Test the host-side attestation verifier."""

    def test_verify_matching_pcrs(self) -> None:
        expected = {
            0: "aabbccdd" * 12,  # 96 hex chars = 48 bytes
            1: "11223344" * 12,
            2: "55667788" * 12,
        }
        verifier = AttestationVerifier(expected_pcrs=expected)
        result = verifier.verify_pcrs(expected)
        assert result.valid is True
        assert result.image_hash_match is True

    def test_verify_mismatched_pcrs(self) -> None:
        expected = {0: "aabbccdd" * 12}
        actual = {0: "11111111" * 12}
        verifier = AttestationVerifier(expected_pcrs=expected)
        result = verifier.verify_pcrs(actual)
        assert result.valid is False
        assert "mismatch" in result.error.lower()

    def test_verify_no_expected_configured(self) -> None:
        verifier = AttestationVerifier()  # No expected PCRs
        result = verifier.verify_pcrs({0: "abc123"})
        assert result.valid is False
        assert "No expected" in result.error

    def test_verify_debug_mode_detected(self) -> None:
        expected = {0: "aabbccdd" * 12}
        zero_pcrs = {0: "0" * 96, 1: "0" * 96, 2: "0" * 96}
        verifier = AttestationVerifier(expected_pcrs=expected)
        result = verifier.verify_pcrs(zero_pcrs)
        assert result.valid is False
        assert "debug" in result.error.lower()

    def test_verify_missing_pcr(self) -> None:
        expected = {0: "aabbccdd" * 12, 1: "11223344" * 12}
        actual = {0: "aabbccdd" * 12}  # Missing PCR[1]
        verifier = AttestationVerifier(expected_pcrs=expected)
        result = verifier.verify_pcrs(actual)
        assert result.valid is False
        assert "missing" in result.error.lower()

    def test_verify_document_json(self) -> None:
        import json
        pcr_hex = "aabbccdd" * 12
        doc = json.dumps({
            "pcrs": {"0": pcr_hex, "1": pcr_hex, "2": pcr_hex},
            "certificate": "mock",
        }).encode()

        verifier = AttestationVerifier(expected_pcrs={
            0: pcr_hex, 1: pcr_hex, 2: pcr_hex,
        })
        result = verifier.verify_document(doc)
        assert result.valid is True

    def test_verify_task_integrity_match(self) -> None:
        pcr_hex = "aabbccdd" * 12
        verifier = AttestationVerifier(expected_pcrs={0: pcr_hex})
        receipt = {
            "task_hash": "abc123",
            "pcrs": {"0": pcr_hex},
        }
        assert verifier.verify_task_integrity("abc123", receipt) is True

    def test_verify_task_integrity_hash_mismatch(self) -> None:
        pcr_hex = "aabbccdd" * 12
        verifier = AttestationVerifier(expected_pcrs={0: pcr_hex})
        receipt = {
            "task_hash": "wrong_hash",
            "pcrs": {"0": pcr_hex},
        }
        assert verifier.verify_task_integrity("abc123", receipt) is False

    def test_verify_invalid_document(self) -> None:
        verifier = AttestationVerifier()
        result = verifier.verify_document(b"not valid at all \xff\xfe")
        assert result.valid is False
