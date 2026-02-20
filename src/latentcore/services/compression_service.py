from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch

from latentcore.engine.text_decoder import TextDecoder
from latentcore.engine.text_encoder import TextEncoder
from latentcore.engine.vq_codebook import VectorQuantizer
from latentcore.storage.latent_store import LatentStore

logger = logging.getLogger("latentcore.compression")


@dataclass
class CompressResult:
    ref_id: str
    indices: list[int]
    num_vectors: int
    original_token_count: int
    compression_ratio: float


class CompressionService:
    """Orchestrates text compression through the VQ pipeline.

    Flow: text -> TextEncoder -> VQ encode -> store indices -> ref_id
    Reverse: ref_id -> load indices -> VQ decode -> TextDecoder -> text
    """

    def __init__(
        self,
        vq_engine: VectorQuantizer,
        text_encoder: TextEncoder,
        text_decoder: TextDecoder,
        store: LatentStore,
        infini_engine=None,
    ):
        self.vq_engine = vq_engine
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.store = store
        self.infini_engine = infini_engine

    async def compress(self, text: str, metadata: dict | None = None) -> CompressResult:
        """Compress text to VQ indices and store persistently.

        Args:
            text: Input text to compress.
            metadata: Optional metadata to store alongside the indices.

        Returns:
            CompressResult with ref_id, indices, and compression metrics.
        """
        # Encode text to embeddings (CPU/GPU bound -> run in thread)
        embeddings = await asyncio.to_thread(self.text_encoder.encode, text)
        original_token_count = self.text_encoder.token_count(text)

        # VQ encode to discrete indices
        quantized, indices_tensor = await asyncio.to_thread(self.vq_engine.encode, embeddings)
        indices = indices_tensor.squeeze(0).tolist()

        # Update infini-attention memory if available
        if self.infini_engine is not None:
            await asyncio.to_thread(self.infini_engine.update, quantized, quantized)

        # Store in database
        num_vectors = len(indices)
        compression_ratio = original_token_count / num_vectors if num_vectors > 0 else 0.0

        ref_id = await self.store.save_ref(
            indices=indices,
            original_token_count=original_token_count,
            metadata=metadata,
        )

        logger.info(
            "Compressed %d tokens -> %d indices (%.1fx) ref=%s",
            original_token_count, num_vectors, compression_ratio, ref_id,
        )

        return CompressResult(
            ref_id=ref_id,
            indices=indices,
            num_vectors=num_vectors,
            original_token_count=original_token_count,
            compression_ratio=compression_ratio,
        )

    async def decompress(
        self, ref_id: str | None = None, indices: list[int] | None = None
    ) -> str:
        """Decompress VQ indices back to text.

        Either ref_id (loads from DB) or raw indices must be provided.
        """
        if ref_id is not None:
            ref_data = await self.store.load_ref(ref_id)
            if ref_data is None:
                raise ValueError(f"Latent ref not found: {ref_id}")
            indices = ref_data["indices"]

        if indices is None:
            raise ValueError("Either ref_id or indices must be provided")

        # VQ decode indices to embeddings
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        if indices_tensor.dim() == 1:
            indices_tensor = indices_tensor.unsqueeze(0)  # [1, seq_len]

        embeddings = await asyncio.to_thread(self.vq_engine.decode, indices_tensor)

        # Decode embeddings to text
        text = await asyncio.to_thread(self.text_decoder.decode, embeddings)

        return text

    async def resolve_latent_refs(self, ref_ids: list[str]) -> torch.Tensor:
        """Resolve multiple VQ refs and return combined embedding tensor.

        Used for injecting historical context into the inference pipeline.
        """
        all_embeddings = []
        for ref_id in ref_ids:
            ref_data = await self.store.load_ref(ref_id)
            if ref_data is None:
                logger.warning("Latent ref not found: %s (skipping)", ref_id)
                continue

            indices_tensor = torch.tensor(ref_data["indices"], dtype=torch.long).unsqueeze(0)
            embeddings = self.vq_engine.decode(indices_tensor)
            all_embeddings.append(embeddings)

        if not all_embeddings:
            return torch.empty(0)

        return torch.cat(all_embeddings, dim=1)  # [1, total_seq_len, embedding_dim]
