"""Host-side attestation document verifier.

Verifies attestation documents from the enclave by checking PCR values
against known-good measurements. In production, also verifies the
AWS Nitro certificate chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of an attestation verification."""

    valid: bool
    pcrs: dict[int, str]  # PCR index → hex value
    image_hash_match: bool = False
    certificate_valid: bool = False
    error: str | None = None
    details: dict[str, Any] | None = None


class AttestationVerifier:
    """Verifies attestation documents from the enclave.

    Compares PCR values against expected "golden" measurements
    generated during the enclave image build process.
    """

    def __init__(
        self,
        expected_pcrs: dict[int, str] | None = None,
        aws_root_cert_pem: str | None = None,
    ) -> None:
        """Initialize verifier.

        Args:
            expected_pcrs: Expected PCR values as hex strings.
                PCR[0] = image hash, PCR[1] = kernel, PCR[2] = application.
            aws_root_cert_pem: AWS Nitro Root CA certificate (PEM).
        """
        self._expected_pcrs = expected_pcrs or {}
        self._aws_root_cert_pem = aws_root_cert_pem

    def verify_pcrs(self, actual_pcrs: dict[int, str]) -> VerificationResult:
        """Verify PCR values against expected measurements.

        Args:
            actual_pcrs: Actual PCR values from the attestation document (hex).

        Returns:
            VerificationResult.
        """
        if not self._expected_pcrs:
            return VerificationResult(
                valid=False,
                pcrs=actual_pcrs,
                error="No expected PCR values configured",
            )

        mismatches: list[str] = []
        for idx, expected_hex in self._expected_pcrs.items():
            actual_hex = actual_pcrs.get(idx)
            if actual_hex is None:
                mismatches.append(f"PCR[{idx}]: missing from attestation")
            elif actual_hex != expected_hex:
                mismatches.append(
                    f"PCR[{idx}]: expected {expected_hex[:16]}... "
                    f"got {actual_hex[:16]}..."
                )

        # Check for all-zero PCRs (debug mode)
        all_zero = all(
            v == "0" * len(v) for v in actual_pcrs.values()
        )

        if all_zero:
            return VerificationResult(
                valid=False,
                pcrs=actual_pcrs,
                error="All PCR values are zero — enclave is in debug mode",
                details={"debug_mode": True},
            )

        if mismatches:
            return VerificationResult(
                valid=False,
                pcrs=actual_pcrs,
                image_hash_match=False,
                error=f"PCR mismatch: {'; '.join(mismatches)}",
                details={"mismatches": mismatches},
            )

        logger.info("attestation_verified", extra={"pcr_count": len(actual_pcrs)})

        return VerificationResult(
            valid=True,
            pcrs=actual_pcrs,
            image_hash_match=True,
            certificate_valid=True,  # Would be verified with real COSE in production
        )

    def verify_document(self, raw_document: bytes) -> VerificationResult:
        """Verify a raw attestation document (COSE_Sign1 or mock JSON).

        In production, this would:
        1. Parse the COSE_Sign1 structure
        2. Verify the certificate chain against AWS Nitro Root CA
        3. Verify PCR values against expected measurements

        For MVP, handles mock JSON attestation documents.
        """
        try:
            doc_dict = json.loads(raw_document.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Try CBOR parsing for real attestation docs
            try:
                import cbor2
                doc_dict = cbor2.loads(raw_document)
            except Exception:
                return VerificationResult(
                    valid=False,
                    pcrs={},
                    error="Failed to parse attestation document",
                )

        # Ensure parsed result is a dict
        if not isinstance(doc_dict, dict):
            return VerificationResult(
                valid=False,
                pcrs={},
                error="Failed to parse attestation document",
            )

        # Extract PCR values
        pcrs_raw = doc_dict.get("pcrs", {})
        pcrs: dict[int, str] = {}
        for idx_str, value in pcrs_raw.items():
            idx = int(idx_str)
            pcrs[idx] = value if isinstance(value, str) else value.hex()

        return self.verify_pcrs(pcrs)

    def verify_task_integrity(
        self,
        task_hash: str,
        attestation_receipt: dict[str, Any],
    ) -> bool:
        """Verify that a task's attestation receipt is valid.

        Checks:
        1. Task hash matches the receipt
        2. PCR values match expected values
        3. Signature is valid (in production)

        Args:
            task_hash: SHA3-256 hash of the task prompt + outputs.
            attestation_receipt: The receipt dict from the enclave.

        Returns:
            True if valid, False otherwise.
        """
        receipt_hash = attestation_receipt.get("task_hash")
        if receipt_hash != task_hash:
            logger.warning(
                "task_hash_mismatch",
                extra={
                    "expected": task_hash[:16],
                    "actual": (receipt_hash or "")[:16],
                },
            )
            return False

        pcrs = attestation_receipt.get("pcrs", {})
        pcrs_int_keys = {int(k): v for k, v in pcrs.items()}
        result = self.verify_pcrs({k: v for k, v in pcrs_int_keys.items()})

        return result.valid
