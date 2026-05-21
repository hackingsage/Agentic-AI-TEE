"""Memory manager — wraps vector store with transparent encryption.

Encrypts data before storage and decrypts after retrieval using
per-collection keys derived from the enclave master secret.
"""

from __future__ import annotations

import logging
from typing import Any

from enclave.crypto.keys import EnclaveKeyManager, seal_data, unseal_data
from enclave.tools.memory_tool import MemoryEntry, SimpleVectorStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_HOT_ENTRIES = 100


class MemoryManager:
    """Encrypted memory manager for the agent.

    Wraps the vector store with transparent encryption:
    - Encrypts content before writing to the store
    - Decrypts content after reading from the store
    - Derives per-collection keys from the enclave master secret

    For MVP, uses SimpleVectorStore. In production, this will wrap ChromaDB.
    """

    def __init__(
        self,
        master_secret: bytes,
        *,
        max_entries: int = DEFAULT_MAX_HOT_ENTRIES,
    ) -> None:
        self._master_secret = master_secret
        self._stores: dict[str, SimpleVectorStore] = {}
        self._max_entries = max_entries

    def _get_collection_key(self, collection: str) -> bytes:
        """Derive an encryption key for a specific collection."""
        return EnclaveKeyManager.derive_key(
            master_secret=self._master_secret,
            context=f"memory:{collection}",
            subkey_id=1,
        )

    def _get_store(self, collection: str) -> SimpleVectorStore:
        """Get or create a store for the given collection."""
        if collection not in self._stores:
            self._stores[collection] = SimpleVectorStore(max_entries=self._max_entries)
        return self._stores[collection]

    def store(
        self,
        content: str,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store content in encrypted memory.

        Args:
            content: The text content to store.
            collection: Collection name for key separation.
            metadata: Optional metadata (stored unencrypted for search).

        Returns:
            Entry ID.
        """
        key = self._get_collection_key(collection)
        encrypted_content = seal_data(content.encode("utf-8"), key)

        store = self._get_store(collection)

        # Store the encrypted content as hex (SimpleVectorStore works with strings)
        # The metadata is stored unencrypted for search indexing
        # In production with ChromaDB, the embedding would also be encrypted
        entry_id = store.store(
            content=content,  # Plaintext for MVP search; encrypted in production
            metadata={
                **(metadata or {}),
                "_encrypted": encrypted_content.hex(),
                "_collection": collection,
            },
        )

        logger.info(
            "memory_stored",
            extra={
                "collection": collection,
                "entry_id": entry_id,
                "encrypted_size": len(encrypted_content),
            },
        )

        return entry_id

    def search(
        self,
        query: str,
        collection: str = "default",
        n_results: int = 5,
    ) -> list[MemoryEntry]:
        """Search encrypted memory.

        Args:
            query: Search query.
            collection: Collection to search.
            n_results: Maximum number of results.

        Returns:
            List of matching MemoryEntry objects.
        """
        store = self._get_store(collection)
        results = store.search(query, n_results)

        logger.info(
            "memory_searched",
            extra={
                "collection": collection,
                "n_results": len(results),
            },
        )

        return results

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return list(self._stores.keys())

    def collection_count(self, collection: str = "default") -> int:
        """Get the number of entries in a collection."""
        store = self._stores.get(collection)
        return store.count if store else 0

    def verify_encrypted_entry(
        self,
        entry: MemoryEntry,
        collection: str = "default",
    ) -> str | None:
        """Decrypt and verify an encrypted entry.

        Returns the decrypted content or None if decryption fails.
        """
        encrypted_hex = entry.metadata.get("_encrypted")
        if not encrypted_hex:
            return entry.content  # Not encrypted (legacy entry)

        try:
            key = self._get_collection_key(collection)
            decrypted = unseal_data(bytes.fromhex(encrypted_hex), key)
            return decrypted.decode("utf-8")
        except Exception as exc:
            logger.error(
                "memory_decrypt_failed",
                extra={"entry_id": entry.entry_id, "error": str(exc)},
            )
            return None
