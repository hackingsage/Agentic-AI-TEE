# 🔒 Enclave — TEE-Protected AI Agent Platform

> **Your code, your data, your prompts — provably private. The AI helps you, but never learns from you.**

Enclave is a privacy-first autonomous AI agent platform where **all LLM inference and tool execution happens inside a Trusted Execution Environment (TEE)**. It cryptographically proves that user data, prompts, and task context never leave the secure enclave in plaintext.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│           Next.js frontend  ←→  WebAuthn auth               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS + JWT
┌────────────────────────▼────────────────────────────────────┐
│                     GATEWAY LAYER (EC2 host)                 │
│   FastAPI  ─── SSE streaming ─── Task queue                  │
│   Attestation verifier        Audit log (append-only)        │
└──────────────vsock─────────────────────────────────────────-─┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   ★ NITRO ENCLAVE ★                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             Agent Controller (Python)               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │    │
│  │  │  Planner │  │ Executor │  │  Memory Manager   │ │    │
│  │  │ (Claude) │  │(sandbox) │  │ (sealed storage)  │ │    │
│  │  └────┬─────┘  └────┬─────┘  └────────┬──────────┘ │    │
│  │       │             │                  │            │    │
│  │  ┌────▼─────────────▼──────────────────▼──────────┐ │    │
│  │  │              Tool Router                        │ │    │
│  │  │  code_exec │ file_ops │ browser │ api_call      │ │    │
│  │  └────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│  Enclave crypto keys (not accessible from host process)     │
└──────────────vsock─────────────────────────────────────────-┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  PRIVACY PROXY (Go)                          │
│   Key rotation │ Metadata stripping │ Dummy-request padding │
└──────────────mTLS──────────────────────────────────────────-┘
                         │
              LLM Provider API (Anthropic / OpenAI)
```

## Privacy Guarantees

| Guarantee | How It Works |
|---|---|
| **TEE-enforced inference** | LLM API calls made from inside an attested AWS Nitro Enclave. The host cannot observe plaintext. |
| **Attestation receipts** | Every task produces a signed attestation report proving code version, enclave identity, and data integrity. |
| **Zero-training** | API calls route through a privacy proxy that strips metadata and rotates API keys. |
| **Sealed memory** | Agent state encrypted with enclave-only keys derived from PCR measurements. Tampered enclaves lose access. |
| **Task integrity** | SHA3-256 hash of every task's prompt + output included in the attestation receipt. |

## Quick Start

### Prerequisites
- Python 3.12+
- Go 1.22+ (for the privacy proxy)

### Setup

```bash
# Clone and enter the project
cd TEE

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Run the Interactive TUI (Recommended)

Enclave includes a terminal interactive UI (TUI) that features real-time **Claude Code-style token and event streaming** (thinking steps, live text tokens, tool calls, and results) from the secure enclave:

```bash
source .venv/bin/activate
python tui.py
```

Inside the TUI, you can submit tasks, switch models/providers, inspect history, verify attestation receipts, and observe the agent's reasoning process in real-time.

### Run the Host API (Development Gateway)

```bash
source .venv/bin/activate
uvicorn host.api.main:app --reload --port 8000
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Submit a new task |
| `GET` | `/tasks/{id}` | Get task status |
| `GET` | `/tasks/{id}/stream` | SSE stream of task events |
| `GET` | `/tasks/{id}/attestation` | Get attestation receipt |
| `GET` | `/enclave/attest` | Get enclave attestation document |
| `GET` | `/health` | Health check |

## Project Structure

```
TEE/
├── enclave/              # Runs inside the TEE
│   ├── agent/            # AgentController, Planner, LLM client
│   │   ├── controller.py # Main agent step loop (native tool calling)
│   │   ├── planner.py    # System prompt builder (native tool use parameters)
│   │   ├── llm_client.py # Provider clients (Anthropic with streaming, OpenAI, etc.)
│   │   └── models.py     # All dataclasses (Message, ContentBlock, StepEvent, etc.)
│   ├── tools/            # Tool implementations
│   │   ├── base.py       # BaseTool ABC + ToolRegistry (JSON tool definitions)
│   │   ├── router.py     # ToolRouter (dispatch + validation)
│   │   ├── code_executor.py
│   │   ├── file_ops.py
│   │   ├── browser_tool.py
│   │   ├── api_call.py
│   │   └── memory_tool.py
│   ├── crypto/           # Cryptographic primitives
│   │   ├── keys.py       # Key generation, HKDF, SecretBox
│   │   ├── attestation.py # Nitro + Mock attestation providers
│   │   └── sealing.py    # PCR-bound secret sealing
│   ├── memory/           # Persistent encrypted state
│   │   ├── manager.py    # MemoryManager (encrypted ChromaDB wrapper)
│   │   └── state_db.py   # TaskStateDB (async SQLite)
│   └── vsock/            # Enclave ↔ Host communication
│       ├── protocol.py   # Length-prefix message framing
│       ├── server.py     # Async server (passes connection writer for streaming)
│       └── client.py     # Async client (synchronous call & stream-iterator send)
├── host/                 # Runs outside the TEE (untrusted)
│   ├── api/main.py       # FastAPI gateway
│   └── attestation/      # Host-side attestation verifier
├── proxy/                # Go privacy proxy
├── tests/                # Full test suite (114 tests)
├── docs/                 # Security whitepaper, guides
└── pyproject.toml        # Project configuration
```

## Security Model

### Threat Model
- **Curious cloud provider**: Cannot read Nitro Enclave memory
- **Compromised host**: Can relay messages but cannot decrypt them
- **LLM provider**: Cannot see plaintext prompts (privacy proxy strips metadata)
- **Network eavesdropper**: All external traffic is mTLS
- **Malicious LLM responses**: All tool calls sandboxed, validated against schema

### Cryptographic Stack
- **Signing**: Ed25519 (via libsodium/PyNaCl)
- **Encryption**: XSalsa20-Poly1305 (NaCl SecretBox)
- **Key exchange**: Curve25519 (NaCl SealedBox)
- **Key derivation**: HKDF-SHA256
- **Attestation**: AWS Nitro Attestation Document (COSE_Sign1 + CBOR)
- **Task integrity**: SHA3-256

### The North Star Test
> *"If Anthropic, AWS, or a government agency subpoenaed all logs, host process memory, and network traffic, what would they learn about the user's tasks?"*

**Answer: Nothing useful.** They would see encrypted blobs, metadata-stripped API calls, and an attestation document proving legitimate code was running.

## Development

### Running Tests
```bash
pytest tests/ -v                              # All tests
pytest tests/ --cov=enclave --cov-report=term  # With coverage
pytest tests/test_agent_loop.py -v            # Agent loop only
pytest tests/test_attestation.py -v           # Crypto + attestation
```

### Code Quality
```bash
ruff check enclave/ host/ tests/     # Linting
mypy enclave/ host/                   # Type checking
bandit -r enclave/ host/ -ll          # Security scanning
```

## License

Private — All rights reserved.
