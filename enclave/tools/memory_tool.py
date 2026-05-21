"""Memory search tool — semantic search over past tasks and stored knowledge.

Uses a simple in-memory vector store for MVP. ChromaDB integration
planned for production.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

VALID_OPERATIONS = {"search", "store", "list_recent"}


@dataclass
class MemoryEntry:
    """A single entry in the memory store."""

    entry_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class SimpleVectorStore:
    """In-memory vector store using basic text similarity.

    This is the MVP implementation. In production, this will be replaced
    with ChromaDB backed by FAISS, with transparent encryption.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory entry. Returns the entry ID."""
        entry_id = hashlib.sha256(
            f"{content}{time.time()}".encode()
        ).hexdigest()[:16]

        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            metadata=metadata or {},
        )
        self._entries.append(entry)

        # Evict oldest entries if over capacity
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        return entry_id

    def search(self, query: str, n_results: int = 5) -> list[MemoryEntry]:
        """Search for similar entries using basic keyword matching.

        MVP implementation uses word overlap scoring.
        Production will use proper embedding-based similarity.
        """
        query_words = set(query.lower().split())

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            entry_words = set(entry.content.lower().split())
            if not query_words or not entry_words:
                continue
            # Jaccard similarity
            intersection = len(query_words & entry_words)
            union = len(query_words | entry_words)
            score = intersection / union if union > 0 else 0.0
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n_results]]

    def list_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most recent entries."""
        return list(reversed(self._entries[-n:]))

    @property
    def count(self) -> int:
        return len(self._entries)


class MemoryTool(BaseTool):
    """Semantic search and storage over agent memory.

    Allows the agent to store observations, search past context,
    and retrieve recent entries.
    """

    name = "memory_search"
    description = (
        "Search and store information in the agent's long-term memory. "
        "Use 'store' to save important observations, facts, or code snippets for later use. "
        "Use 'search' to find relevant past information. "
        "Use 'list_recent' to see the most recent stored entries."
    )

    def __init__(self, store: SimpleVectorStore | None = None) -> None:
        self._store = store or SimpleVectorStore()

    @property
    def store(self) -> SimpleVectorStore:
        return self._store

    def validate_args(self, args: dict[str, Any]) -> str | None:
        operation = args.get("operation")
        if not operation:
            return "Missing required argument: 'operation'"
        if operation not in VALID_OPERATIONS:
            return f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
        if operation == "search" and not args.get("query"):
            return "Missing required argument: 'query' for search operation"
        if operation == "store" and not args.get("content"):
            return "Missing required argument: 'content' for store operation"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        operation: str = kwargs["operation"]

        if operation == "store":
            content: str = kwargs["content"]
            metadata: dict[str, Any] = kwargs.get("metadata", {})
            entry_id = self._store.store(content, metadata)
            return ToolOutput(
                success=True,
                result=f"Stored entry '{entry_id}'. Total entries: {self._store.count}",
            )

        elif operation == "search":
            query: str = kwargs["query"]
            n_results: int = kwargs.get("n_results", 5)
            results = self._store.search(query, n_results)
            if not results:
                return ToolOutput(success=True, result="No matching entries found.")

            formatted = []
            for entry in results:
                formatted.append(
                    f"[{entry.entry_id}] ({entry.metadata})\n{entry.content}"
                )
            return ToolOutput(success=True, result="\n---\n".join(formatted))

        elif operation == "list_recent":
            n: int = kwargs.get("n", 10)
            entries = self._store.list_recent(n)
            if not entries:
                return ToolOutput(success=True, result="No entries in memory.")

            formatted = []
            for entry in entries:
                formatted.append(
                    f"[{entry.entry_id}] {entry.content[:100]}..."
                )
            return ToolOutput(success=True, result="\n".join(formatted))

        return ToolOutput(success=False, error=f"Unknown operation: {operation}")

    def schema_xml(self) -> str:
        return """<tool name="memory_search">
  <description>Search and store information in the agent's long-term memory.</description>
  <args>
    <arg name="operation" type="string" required="true">One of: "search", "store", "list_recent"</arg>
    <arg name="query" type="string" required="false">Search query (required for "search")</arg>
    <arg name="content" type="string" required="false">Content to store (required for "store")</arg>
    <arg name="metadata" type="object" required="false">Optional metadata for stored entries</arg>
    <arg name="n_results" type="integer" required="false">Number of results to return (default: 5)</arg>
  </args>
</tool>"""
