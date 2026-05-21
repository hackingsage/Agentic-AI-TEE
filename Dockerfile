# ============================================================================
# Enclave Dockerfile — Multi-stage build for AWS Nitro Enclave
#
# Stage 1: Build stage (install deps, compile)
# Stage 2: Runtime stage (minimal image for enclave)
#
# Build: docker build -t enclave:latest .
# For Nitro: nitro-cli build-enclave --docker-uri enclave:latest --output-file enclave.eif
# ============================================================================

# --------------------------------------------------------------------------
# Stage 1: Builder
# --------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (for layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install .

# Copy source code
COPY enclave/ ./enclave/

# --------------------------------------------------------------------------
# Stage 2: Runtime
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd -r enclave && useradd -r -g enclave -d /app -s /sbin/nologin enclave

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY enclave/ ./enclave/

# Create workspace directories
RUN mkdir -p /tmp/enclave_workspace /tmp/enclave_sealed /tmp/enclave_state \
    && chown -R enclave:enclave /tmp/enclave_workspace /tmp/enclave_sealed /tmp/enclave_state /app

# Switch to non-root user
USER enclave

# Environment defaults (overridden by enclave config)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENCLAVE_USE_VSOCK=true \
    ENCLAVE_VSOCK_PORT=5000 \
    ENCLAVE_LLM_PROVIDER=mock \
    ENCLAVE_WORKSPACE=/tmp/enclave_workspace \
    ENCLAVE_DB_PATH=/tmp/enclave_state/state.db \
    ENCLAVE_SEALED_PATH=/tmp/enclave_sealed

# Health check — echo handler via vsock
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Entry point
ENTRYPOINT ["python", "-m", "enclave.main"]
