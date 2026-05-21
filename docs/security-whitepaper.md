# Enclave Security Whitepaper

## Threat Model & Cryptographic Guarantees

**Version 0.1.0** | May 2026

---

## 1. Overview

Enclave is an autonomous AI agent platform that executes all LLM inference and tool operations inside a Trusted Execution Environment (TEE). This document describes the threat model, cryptographic guarantees, known limitations, and how users can independently verify the system's integrity.

## 2. Threat Model

### 2.1 Adversaries

| Adversary | Capability | Mitigation |
|---|---|---|
| **Curious cloud provider** | Can observe host memory, network traffic, disk I/O | Nitro Enclave memory is hardware-isolated; PCIe boundary prevents host access |
| **Compromised host process** | Can intercept vsock messages, modify relay logic | Host only sees encrypted blobs; cannot forge attestation documents |
| **LLM provider** | Can observe API request content and train on it | Privacy proxy strips metadata, rotates keys; requests are disassociated from user identity |
| **Network eavesdropper** | Can observe packet sizes and timing | mTLS for all external traffic; optional dummy-request padding for traffic analysis resistance |
| **Malicious LLM responses** | LLM could hallucinate dangerous tool calls | All tool calls validated against schema; code execution in sandboxed subprocess with seccomp |

### 2.2 Trust Assumptions

- **AWS hardware**: We trust that AWS Nitro Enclave hardware correctly enforces memory isolation and attestation signing.
- **Cryptographic primitives**: We trust libsodium's implementations of Ed25519, Curve25519, XSalsa20-Poly1305, and the `cryptography` library's HKDF-SHA256.
- **Enclave image integrity**: Users must verify the attestation document's PCR values match the published, reproducibly-built image hash.

## 3. Cryptographic Guarantees

### 3.1 Key Hierarchy

```
Master Secret (32 bytes, randomly generated or unsealed)
  ├── Ed25519 Signing Key (derived via HKDF, context="ed25519_signing")
  ├── Curve25519 Encryption Key (derived via HKDF, context="curve25519_encryption")
  ├── Per-task encryption keys (derived via HKDF, context="task:{task_id}")
  └── Per-collection memory keys (derived via HKDF, context="memory:{collection}")
```

- All keys derived from a single master secret using HKDF-SHA256
- Different contexts ensure cryptographic separation between key types
- Master secret sealed to the enclave's PCR measurements

### 3.2 Attestation

Every task produces an **attestation receipt** containing:
1. **PCR values**: SHA-384 hashes of the enclave image (PCR[0]), kernel (PCR[1]), and application (PCR[2])
2. **Task hash**: SHA3-256 of `(task_description + all_tool_outputs + final_response)`
3. **Enclave signature**: Ed25519 signature over `(task_hash + attestation_document)`
4. **Timestamp**: When the task completed

Users can verify that:
- The enclave was running the exact code they audited (PCR[0] match)
- The task output was produced by that code (task hash in signed attestation)
- No data left the enclave unencrypted (architectural guarantee of Nitro Enclaves)

### 3.3 Key Sealing

The master secret is encrypted with a key derived from PCR measurements:

```
pcr_material = PCR[0] || PCR[1] || PCR[2]  (concatenated in order)
sealing_key = HKDF-SHA256(SHA-256(pcr_material), context="sealed_secret_store")
sealed_secret = SecretBox.encrypt(master_secret, sealing_key)
```

If the enclave image is modified (compromised), PCR values change, the sealing key changes, and the master secret becomes unreadable.

### 3.4 Data At Rest

All persisted data is encrypted before leaving the enclave:
- **Task state**: SQLite database (SQLCipher in production, AES-256)
- **Memory entries**: Encrypted with per-collection keys (XSalsa20-Poly1305)
- **File workspace**: Per-task encrypted workspace (AES-256-GCM)

## 4. Privacy Proxy

The Go privacy proxy sits between the enclave and the LLM API provider:

| Feature | Implementation |
|---|---|
| Metadata stripping | Removes `User-Agent`, `X-Forwarded-For`, `X-Real-IP`, cookies, correlation IDs |
| API key rotation | Round-robin key ring prevents request correlation |
| No-train headers | Adds `Anthropic-No-Train: true` to all requests |
| Dummy padding | Optional synthetic requests with realistic timing to prevent traffic analysis |
| No content logging | Logs only timestamps, byte counts, and status codes |

The proxy **does NOT**:
- Decrypt request or response content
- Log request bodies
- Maintain persistent state about request content

## 5. Known Limitations

1. **Side-channel attacks**: Nitro Enclaves provide strong isolation but are not immune to all side-channel attacks (e.g., timing attacks on shared CPU caches). We mitigate by using constant-time crypto (libsodium) and plan `MADV_WIPEONFORK` for sensitive memory regions.

2. **Traffic analysis**: Even with dummy padding, an attacker who can observe exact packet sizes and timing patterns may infer some information about task complexity. This is a fundamental limitation — we reduce but do not eliminate this risk.

3. **LLM provider cooperation**: The `Anthropic-No-Train` header is a contractual control, not a technical guarantee. We trust that the LLM provider honors it. For maximum protection, use Ollama for fully air-gapped inference.

4. **Key migration**: Updating the enclave image changes PCR values, making sealed secrets unreadable. A key migration ceremony is required: the old enclave exports a migration bundle encrypted to the new image's expected public key.

5. **Docker-in-Docker**: Code execution inside the enclave currently uses subprocess isolation. Full Docker-in-Docker with seccomp profiles is planned for production.

## 6. How to Verify Attestation

See [attestation-guide.md](attestation-guide.md) for the step-by-step user verification process.

```bash
# Quick verification using the provided script
./scripts/verify-attestation.sh <task_id> <expected_pcr0_hex>
```
