# Attestation Verification Guide

## How to Verify an Enclave Task's Integrity

This guide explains how to independently verify that a task was executed inside a legitimate, unmodified Enclave instance.

## What You're Verifying

When you verify an attestation receipt, you confirm three things:

1. **Code integrity**: The enclave was running the exact code version you expect (PCR[0] matches the published image hash)
2. **Output integrity**: The task output was produced by that code (task hash is included in the signed attestation)
3. **Privacy guarantee**: The enclave architecture ensures no plaintext data left the enclave

## Step 1: Get the Attestation Receipt

```bash
# After a task completes, fetch its attestation receipt
curl https://your-enclave-host/tasks/{task_id}/attestation
```

Response:
```json
{
  "task_id": "abc123def456",
  "task_hash": "a1b2c3d4...",
  "pcrs": {
    "0": "sha384-hex-of-image-hash...",
    "1": "sha384-hex-of-kernel...",
    "2": "sha384-hex-of-application..."
  },
  "timestamp": 1716100000.0,
  "signature": "ed25519-signature-hex..."
}
```

## Step 2: Verify PCR Values

Compare PCR[0] against the published image hash for the Enclave version:

```bash
# Published PCR values are available at:
# https://github.com/your-org/enclave/releases/latest/pcrs.json

# Compare
EXPECTED_PCR0="published-pcr0-value"
ACTUAL_PCR0=$(curl -s .../attestation | jq -r '.pcrs["0"]')

if [ "$EXPECTED_PCR0" = "$ACTUAL_PCR0" ]; then
    echo "✅ PCR[0] matches — enclave image is authentic"
else
    echo "❌ PCR[0] MISMATCH — enclave image may be tampered"
fi
```

## Step 3: Verify Task Hash

Recompute the task hash from the prompt and outputs you received:

```python
import hashlib

# Concatenate: task_description + all tool outputs + final response
content = task_description + "\n".join(tool_outputs) + final_response
expected_hash = hashlib.sha3_256(content.encode()).hexdigest()

assert expected_hash == receipt["task_hash"], "Task hash mismatch!"
```

## Step 4: Verify the Signature

```python
from nacl.signing import VerifyKey

# The enclave's public verify key is published with each release
verify_key = VerifyKey(bytes.fromhex(PUBLISHED_VERIFY_KEY))

# Verify the signature covers the task hash + attestation doc
signed_data = bytes.fromhex(receipt["signature"])
verify_key.verify(signed_data)  # Raises if invalid
```

## Automated Verification

Use the provided script:

```bash
./scripts/verify-attestation.sh <task_id> <expected_pcr0>
```

## Red Flags

- **All-zero PCR values**: The enclave is running in debug mode — data is NOT protected
- **PCR[0] mismatch**: The enclave image has been modified — do not trust outputs
- **Missing signature**: The attestation is incomplete — outputs cannot be verified
- **Timestamp far in the past**: The receipt may have been replayed from an old, compromised version
