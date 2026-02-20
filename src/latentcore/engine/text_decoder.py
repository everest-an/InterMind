from __future__ import annotations

import torch
import torch.nn as nn
import tiktoken


class TextDecoder(nn.Module):
    """Reconstructs text from VQ-decoded embeddings.

    Uses a linear projection from embedding_dim to vocab_size followed by
    greedy argmax decoding through tiktoken.

    Note: Reconstruction is lossy. The primary value of VQ encoding is for
    context injection into the LLM's attention mechanism, not for
    human-readable reconstruction.
    """

    def __init__(
        self,
        vocab_size: int = 100277,
        embedding_dim: int = 768,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.projection = nn.Linear(embedding_dim, vocab_size)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return self.projection.weight.device

    @torch.no_grad()
    def decode(self, embeddings: torch.Tensor) -> str:
        """Decode embedding tensor back to text.

        Args:
            embeddings: Tensor of shape [1, seq_len, embedding_dim] or [seq_len, embedding_dim].

        Returns:
            Reconstructed text string (lossy).
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)  # [seq_len, embedding_dim]

        logits = self.projection(embeddings)  # [seq_len, vocab_size]
        token_ids = logits.argmax(dim=-1).tolist()

        # Filter out invalid token IDs
        valid_ids = [tid for tid in token_ids if 0 <= tid < self.vocab_size]
        try:
            return self._tokenizer.decode(valid_ids)
        except Exception:
            return ""
