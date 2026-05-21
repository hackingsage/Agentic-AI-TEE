"""FastAPI host gateway — serves as the untrusted relay between users and the enclave.

The host can relay messages but cannot read encrypted content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

# --- Pydantic Models ---


class TaskSubmission(BaseModel):
    """Request body for task submission."""

    description: str = Field(..., min_length=1, max_length=50000)
    budget_usd: float = Field(default=5.0, ge=0.01, le=100.0)
    max_steps: int = Field(default=50, ge=1, le=200)
    timeout_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    tool_allowlist: list[str] | None = None
    domain_allowlist: list[str] | None = None


class TaskResponse(BaseModel):
    """Response after task submission."""

    task_id: str
    status: str = "queued"
    message: str = "Task submitted successfully"


class AttestationResponse(BaseModel):
    """Attestation verification response."""

    valid: bool
    pcrs: dict[str, str]
    image_hash_match: bool = False
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    enclave_connected: bool = False
    version: str = "0.1.0"
    uptime_seconds: float = 0.0


# --- Application Setup ---

_start_time = time.time()

# In-memory task store (replaced by enclave StateDB in production)
_tasks: dict[str, dict[str, Any]] = {}
_task_events: dict[str, asyncio.Queue] = {}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Enclave Gateway API",
        description=(
            "Privacy-first AI agent gateway. All task execution happens inside "
            "a Trusted Execution Environment. This API relays requests to the "
            "enclave but cannot read encrypted task content."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Health ---

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            enclave_connected=False,  # Will check vsock in production
            uptime_seconds=round(time.time() - _start_time, 2),
        )

    # --- Tasks ---

    @app.post("/tasks", response_model=TaskResponse)
    async def submit_task(submission: TaskSubmission) -> TaskResponse:
        """Submit a new task for the enclave agent."""
        task_id = uuid.uuid4().hex[:16]

        task_record = {
            "task_id": task_id,
            "description": submission.description,
            "budget_usd": submission.budget_usd,
            "max_steps": submission.max_steps,
            "timeout_seconds": submission.timeout_seconds,
            "status": "queued",
            "created_at": time.time(),
            "result": None,
        }

        _tasks[task_id] = task_record
        _task_events[task_id] = asyncio.Queue()

        logger.info(
            "task_submitted",
            extra={"task_id": task_id, "budget_usd": submission.budget_usd},
        )

        # In production: send task to enclave via vsock
        # For now: acknowledge the queue

        return TaskResponse(task_id=task_id)

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        """Get task status and result."""
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.get("/tasks/{task_id}/stream")
    async def stream_task(task_id: str, request: Request) -> EventSourceResponse:
        """Stream task events via Server-Sent Events."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        event_queue = _task_events.get(task_id)
        if not event_queue:
            raise HTTPException(status_code=404, detail="No event stream for task")

        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    yield {
                        "event": event.get("event_type", "message"),
                        "data": json.dumps(event),
                    }
                    if event.get("event_type") in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": "keepalive"}

        return EventSourceResponse(event_generator())

    @app.get("/tasks/{task_id}/attestation", response_model=AttestationResponse)
    async def get_attestation(task_id: str) -> AttestationResponse:
        """Get attestation receipt for a completed task."""
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task is {task['status']}, attestation only available for completed tasks",
            )

        # In production: fetch attestation from enclave
        return AttestationResponse(
            valid=True,
            pcrs={"0": "mock", "1": "mock", "2": "mock"},
            image_hash_match=True,
        )

    @app.get("/tasks")
    async def list_tasks(
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all tasks."""
        all_tasks = list(_tasks.values())
        all_tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        return {
            "tasks": all_tasks[offset : offset + limit],
            "total": len(all_tasks),
        }

    # --- Attestation ---

    @app.get("/enclave/attest")
    async def enclave_attestation() -> dict[str, Any]:
        """Get the enclave's current attestation document."""
        # In production: forward to enclave via vsock
        return {
            "status": "mock",
            "message": "Attestation not available — no enclave connected",
            "pcrs": {},
        }

    return app


# Create the application instance
app = create_app()
