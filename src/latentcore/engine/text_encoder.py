from __future__ import annotations

import torch
import torch.nn as nn
import tiktoken


class TextEncoder(nn.Module):
    """Converts text to continuous embeddings suitable for VQ encoding.

    Uses tiktoken for tokenization, then a learned embedding table plus
    positional encoding to produce [1, seq_len, embedding_dim] tensors.

    For the MVP this is a lightweight trainable encoder. In production,
    this should be replaced by extracting hidden states from layers -2 to -4
    of the actual LLM (the LatentMAS approach).
    """

    def __init__(
        self,
        vocab_size: int = 100277,
        embedding_dim: int = 768,
        max_seq_len: int = 8192,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to continuous embedding tensor.

        Args:
            text: Input text string.

        Returns:
            Tensor of shape [1, seq_len, embedding_dim].
        """
        token_ids = self._tokenizer.encode(text)
        token_ids = token_ids[: self.max_seq_len]

        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(len(token_ids), dtype=torch.long, device=self.device)

        embeddings = self.token_embedding(ids_tensor) + self.position_embedding(positions)
        return self.layer_norm(embeddings).unsqueeze(0)  # [1, seq_len, embedding_dim]
