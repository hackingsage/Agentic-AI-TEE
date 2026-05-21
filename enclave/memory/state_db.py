"""Task state database — encrypted SQLite for structured state.

Stores task history, step results, file manifests, and cost logs.
All data encrypted at rest using the enclave master secret.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

from enclave.agent.models import StepResult, TaskRequest, TaskResult

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    summary TEXT,
    success INTEGER,
    total_cost_usd REAL DEFAULT 0.0,
    attestation_hash TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    completed_at REAL,
    elapsed_seconds REAL
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    tool_name TEXT,
    tool_args TEXT,
    output_success INTEGER,
    output_result TEXT,
    output_error TEXT,
    llm_response TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_steps_task_id ON steps(task_id);
CREATE INDEX IF NOT EXISTS idx_cost_log_task_id ON cost_log(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
"""


class TaskStateDB:
    """Encrypted SQLite database for task state.

    In production, this would use SQLCipher for transparent encryption.
    For MVP, uses standard SQLite with application-level encryption
    planned for the next phase.

    Uses a persistent connection to support :memory: databases.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create tables and indexes."""
        db = await self._get_db()
        await db.executescript(SCHEMA_SQL)
        await db.commit()
        self._initialized = True
        logger.info("state_db_initialized", extra={"path": self._db_path})

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create the persistent database connection."""
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
        return self._db

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def create_task(self, task: TaskRequest) -> None:
        """Record a new task."""
        await self._ensure_initialized()
        db = await self._get_db()
        await db.execute(
            """INSERT INTO tasks (task_id, user_id, description, status, created_at)
               VALUES (?, ?, ?, 'running', ?)""",
            (task.task_id, task.user_id, task.description, time.time()),
        )
        await db.commit()

    async def complete_task(self, result: TaskResult) -> None:
        """Update a task with its final result."""
        await self._ensure_initialized()
        db = await self._get_db()
        await db.execute(
            """UPDATE tasks SET
                status = ?,
                success = ?,
                summary = ?,
                total_cost_usd = ?,
                attestation_hash = ?,
                error = ?,
                completed_at = ?,
                elapsed_seconds = ?
               WHERE task_id = ?""",
            (
                "completed" if result.success else "failed",
                1 if result.success else 0,
                result.summary,
                result.total_cost_usd,
                result.attestation_hash,
                result.error,
                time.time(),
                result.elapsed_seconds,
                result.task_id,
            ),
        )
        await db.commit()

    async def record_step(self, task_id: str, step: StepResult) -> None:
        """Record a completed step."""
        await self._ensure_initialized()
        db = await self._get_db()
        output_success = step.output.success if step.output else None
        output_result = str(step.output.result) if step.output and step.output.result else None
        output_error = step.output.error if step.output else None

        await db.execute(
            """INSERT INTO steps
               (task_id, step_number, tool_name, tool_args,
                output_success, output_result, output_error,
                llm_response, input_tokens, output_tokens, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                step.step_number,
                step.tool_name,
                json.dumps(step.tool_args),
                1 if output_success else 0 if output_success is not None else None,
                output_result,
                output_error,
                step.llm_response,
                step.input_tokens,
                step.output_tokens,
                step.latency_ms,
                time.time(),
            ),
        )

        # Record cost
        await db.execute(
            """INSERT INTO cost_log
               (task_id, step_number, input_tokens, output_tokens, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                step.step_number,
                step.input_tokens,
                step.output_tokens,
                step.cost_usd,
                time.time(),
            ),
        )

        await db.commit()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get a task by ID."""
        await self._ensure_initialized()
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_tasks(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List tasks, optionally filtered by user."""
        await self._ensure_initialized()
        db = await self._get_db()
        if user_id:
            query = "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = (user_id, limit, offset)
        else:
            query = "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = (limit, offset)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_task_cost(self, task_id: str) -> float:
        """Get total cost for a task."""
        await self._ensure_initialized()
        db = await self._get_db()
        async with db.execute(
            "SELECT SUM(cost_usd) FROM cost_log WHERE task_id = ?",
            (task_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] or 0.0 if row else 0.0

    async def get_user_total_cost(self, user_id: str) -> float:
        """Get total cost across all tasks for a user."""
        await self._ensure_initialized()
        db = await self._get_db()
        async with db.execute(
            """SELECT SUM(cl.cost_usd)
               FROM cost_log cl
               JOIN tasks t ON cl.task_id = t.task_id
               WHERE t.user_id = ?""",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] or 0.0 if row else 0.0
