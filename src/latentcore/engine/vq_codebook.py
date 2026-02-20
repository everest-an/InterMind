from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Maintains a codebook E of shape [num_embeddings, embedding_dim].
    Encodes continuous vectors to nearest codebook indices (L2 distance).
    Decodes indices back to codebook vectors via lookup.

    Uses Exponential Moving Average updates for codebook stability
    and a straight-through estimator for gradient flow.
    """

    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 768,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._ema_decay = ema_decay

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking buffers
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", self.embedding.weight.data.clone())

        # Usage tracking for stats
        self.register_buffer("_usage_count", torch.zeros(num_embeddings, dtype=torch.long))

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def encode(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode continuous vectors to nearest codebook entries.

        Args:
            z_e: Continuous input tensor of shape [batch, seq_len, embedding_dim].

        Returns:
            quantized: Quantized tensor of same shape as z_e.
            indices: Integer indices of shape [batch, seq_len].
        """
        original_shape = z_e.shape
        flat = z_e.reshape(-1, self.embedding_dim)

        # L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e^T
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)

        # Track usage
        self._usage_count.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))

        quantized = self.embedding(indices)
        quantized = quantized.view(original_shape)
        indices = indices.view(original_shape[:-1])
        return quantized, indices

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with straight-through estimator and EMA updates.

        Args:
            z_e: Continuous input tensor of shape [batch, seq_len, embedding_dim].

        Returns:
            quantized_st: Quantized tensor with straight-through gradients.
            indices: Codebook indices.
            loss: Commitment loss scalar.
        """
        quantized, indices = self.encode(z_e)

        if self.training:
            self._ema_update(z_e.detach(), indices)

        # Commitment loss: encourages encoder outputs to stay close to codebook
        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator: copy gradients from quantized to z_e
        quantized_st = z_e + (quantized - z_e).detach()

        return quantized_st, indices, loss

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via Exponential Moving Average."""
        flat = z_e.reshape(-1, self.embedding_dim)
        flat_indices = indices.reshape(-1)

        # One-hot encoding of assignments
        encodings = F.one_hot(flat_indices, self.num_embeddings).float()

        # Update cluster sizes
        self._ema_cluster_size.mul_(self._ema_decay).add_(
            encodings.sum(0), alpha=1 - self._ema_decay
        )

        # Update embedding sums
        dw = encodings.t() @ flat
        self._ema_w.mul_(self._ema_decay).add_(dw, alpha=1 - self._ema_decay)

        # Laplace smoothing to prevent division by zero
        n = self._ema_cluster_size.sum()
        cluster_size = (
            (self._ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )
        self.embedding.weight.data.copy_(self._ema_w / cluster_size.unsqueeze(1))

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode integer indices back to codebook vectors.

        Args:
            indices: Integer tensor of shape [batch, seq_len] or [seq_len].

        Returns:
            Quantized vectors of shape [..., embedding_dim].
        """
        return self.embedding(indices)

    def get_stats(self) -> dict:
        """Return codebook utilization statistics."""
        active = int((self._usage_count > 0).sum().item())
        total_encodes = int(self._usage_count.sum().item())
        return {
            "codebook_size": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "active_codes": active,
            "usage_rate": active / self.num_embeddings if self.num_embeddings > 0 else 0.0,
            "total_encodes": total_encodes,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage counters (e.g., between evaluation epochs)."""
        self._usage_count.zero_()
