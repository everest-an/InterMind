from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger("latentcore.storage")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def init_database(db_path: str) -> aiosqlite.Connection:
    """Initialize the SQLite database with WAL mode and schema."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row

    # Enable WAL mode for better concurrent read performance
    await db.execute("PRAGMA journal_mode=WAL")

    # Apply schema
    schema_sql = _SCHEMA_PATH.read_text()
    await db.executescript(schema_sql)
    await db.commit()

    logger.info("Database initialized at %s", db_path)
    return db
