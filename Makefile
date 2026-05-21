.PHONY: help install test test-cov lint typecheck security run-enclave run-host run-proxy clean

help:  ## Show this help
	@grep -E '^[a-z_-]+:.*## ' Makefile | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

install:  ## Install all dependencies in venv
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------

test:  ## Run all tests
	.venv/bin/python -m pytest tests/ -v

test-cov:  ## Run tests with coverage report
	.venv/bin/python -m pytest tests/ --cov=enclave --cov=host --cov-report=term-missing

test-fast:  ## Run tests without slow tests
	.venv/bin/python -m pytest tests/ -v -x --timeout=10

# --------------------------------------------------------------------------
# Code Quality
# --------------------------------------------------------------------------

lint:  ## Lint with ruff
	.venv/bin/ruff check enclave/ host/ tests/

lint-fix:  ## Auto-fix lint issues
	.venv/bin/ruff check --fix enclave/ host/ tests/

typecheck:  ## Type check with mypy
	.venv/bin/mypy enclave/ host/ --ignore-missing-imports

security:  ## Security scan with bandit
	.venv/bin/bandit -r enclave/ host/ -ll -q

# --------------------------------------------------------------------------
# Run Services
# --------------------------------------------------------------------------

run-enclave:  ## Run the enclave service (local dev mode)
	ENCLAVE_USE_VSOCK=false ENCLAVE_LLM_PROVIDER=mock \
	.venv/bin/python -m enclave.main

run-host:  ## Run the host API gateway
	.venv/bin/uvicorn host.api.main:app --reload --port 8000

run-proxy:  ## Run the Go privacy proxy
	cd proxy && go run main.go

# --------------------------------------------------------------------------
# Docker / Nitro
# --------------------------------------------------------------------------

docker-build:  ## Build the enclave Docker image
	docker build -t enclave:latest .

nitro-build:  ## Build the Nitro Enclave EIF (requires nitro-cli)
	nitro-cli build-enclave --docker-uri enclave:latest --output-file enclave.eif

nitro-run:  ## Run the Nitro Enclave (requires nitro-cli)
	nitro-cli run-enclave --eif-path enclave.eif --memory 4096 --cpu-count 2

nitro-describe:  ## Describe running enclaves
	nitro-cli describe-enclaves

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------

clean:  ## Remove build artifacts
	rm -rf .venv/ build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
