from __future__ import annotations

import numpy as np


def indices_to_bytes(indices: list[int]) -> bytes:
    """Serialize a list of codebook indices to compact binary (int16).

    Supports codebook sizes up to 32767 (int16 max).
    """
    return np.array(indices, dtype=np.int16).tobytes()


def bytes_to_indices(data: bytes) -> list[int]:
    """Deserialize binary blob back to a list of codebook indices."""
    return np.frombuffer(data, dtype=np.int16).tolist()
