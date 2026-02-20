from __future__ import annotations

import json
import logging
import uuid

import aiosqlite

from latentcore.utils.serialization import bytes_to_indices, indices_to_bytes

logger = logging.getLogger("latentcore.storage")


class LatentStore:
    """CRUD operations for the latent_refs table."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def save_ref(
        self,
        indices: list[int],
        original_token_count: int,
        metadata: dict | None = None,
        ref_id: str | None = None,
    ) -> str:
        """Store VQ indices as a compact binary blob.

        Returns:
            The ref_id (UUID) for this latent reference.
        """
        if ref_id is None:
            ref_id = uuid.uuid4().hex

        blob = indices_to_bytes(indices)
        num_indices = len(indices)
        compression_ratio = (
            original_token_count / num_indices if num_indices > 0 else 0.0
        )
        metadata_json = json.dumps(metadata or {})

        await self.db.execute(
            """
            INSERT OR REPLACE INTO latent_refs
            (ref_id, indices_blob, original_token_count, num_indices, compression_ratio, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ref_id, blob, original_token_count, num_indices, compression_ratio, metadata_json),
        )
        await self.db.commit()

        logger.debug("Saved latent ref %s (%d indices)", ref_id, num_indices)
        return ref_id

    async def load_ref(self, ref_id: str) -> dict | None:
        """Load and deserialize a latent ref. Updates access tracking."""
        # Update access tracking first
        await self.db.execute(
            """
            UPDATE latent_refs
            SET accessed_at = datetime('now'), access_count = access_count + 1
            WHERE ref_id = ?
            """,
            (ref_id,),
        )
        await self.db.commit()

        # Then read the updated row
        cursor = await self.db.execute(
            "SELECT * FROM latent_refs WHERE ref_id = ?", (ref_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return {
            "ref_id": row["ref_id"],
            "indices": bytes_to_indices(row["indices_blob"]),
            "original_token_count": row["original_token_count"],
            "num_indices": row["num_indices"],
            "compression_ratio": row["compression_ratio"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "accessed_at": row["accessed_at"],
            "access_count": row["access_count"],
        }

    async def delete_ref(self, ref_id: str) -> bool:
        """Delete a latent ref. Returns True if a row was deleted."""
        cursor = await self.db.execute(
            "DELETE FROM latent_refs WHERE ref_id = ?", (ref_id,)
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def list_refs(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List recent refs ordered by creation time (newest first)."""
        cursor = await self.db.execute(
            """
            SELECT ref_id, original_token_count, num_indices, compression_ratio,
                   metadata_json, created_at, accessed_at, access_count
            FROM latent_refs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "ref_id": row["ref_id"],
                "original_token_count": row["original_token_count"],
                "num_indices": row["num_indices"],
                "compression_ratio": row["compression_ratio"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "accessed_at": row["accessed_at"],
                "access_count": row["access_count"],
            }
            for row in rows
        ]

    async def count_refs(self) -> int:
        """Return the total number of stored latent refs."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM latent_refs")
        row = await cursor.fetchone()
        return row[0]
