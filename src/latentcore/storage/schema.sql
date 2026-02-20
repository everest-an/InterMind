CREATE TABLE IF NOT EXISTS latent_refs (
    ref_id TEXT PRIMARY KEY,
    indices_blob BLOB NOT NULL,
    original_token_count INTEGER NOT NULL,
    num_indices INTEGER NOT NULL,
    compression_ratio REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_latent_refs_created_at ON latent_refs(created_at);
