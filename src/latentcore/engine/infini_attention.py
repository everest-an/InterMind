from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiniAttentionMemory(nn.Module):
    """Fixed-size compressive memory implementing the Infini-attention pattern.

    Maintains a memory matrix M ∈ R^{d_key × d_value} and normalization vector z ∈ R^{d_key}.
    Supports O(1) retrieval and incremental update regardless of sequence history length.

    Based on: "Leave No Context Behind: Efficient Infinite Context Transformers
    with Infini-attention" (Munkhdalai et al., 2024)

    Key operations:
        - retrieve(Q): A_mem = σ(Q) @ M / (σ(Q) @ z), where σ = ELU + 1
        - update(K, V): M += σ(K)^T @ V; z += σ(K).sum(dim=-2)
        - gate(A_mem, A_local): output = sigmoid(β) * A_mem + (1-sigmoid(β)) * A_local
    """

    def __init__(
        self,
        d_key: int = 64,
        d_value: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value

        # Projection layers for mapping input embeddings to key/query space
        self.W_q = nn.Linear(d_value, d_key, bias=False)
        self.W_k = nn.Linear(d_value, d_key, bias=False)

        # Learnable gating parameter per head
        self.beta = nn.Parameter(torch.zeros(1))

        # Compressive memory state (persistent buffers, not parameters)
        self.register_buffer("M", torch.zeros(d_key, d_value))
        self.register_buffer("z", torch.full((d_key,), 1e-8))

        # Statistics tracking
        self.register_buffer("_update_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_total_vectors_stored", torch.tensor(0, dtype=torch.long))

        if device is not None:
            self.to(device)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """σ(x) = ELU(x) + 1, ensures non-negative values for linear attention."""
        return F.elu(x) + 1.0

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve from compressive memory using query vectors.

        Args:
            queries: Tensor of shape [batch, seq_len, d_value].

        Returns:
            Retrieved memory content of shape [batch, seq_len, d_value].
        """
        Q = self.W_q(queries)                         # [B, S, d_key]
        sigma_Q = self._activation(Q)                  # [B, S, d_key]
        numerator = sigma_Q @ self.M                   # [B, S, d_value]
        denominator = sigma_Q @ self.z.unsqueeze(-1)   # [B, S, 1]
        return numerator / (denominator + 1e-8)

    @torch.no_grad()
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Update compressive memory with new key-value pairs.

        Args:
            keys: Tensor of shape [batch, seq_len, d_value].
            values: Tensor of shape [batch, seq_len, d_value].
        """
        K = self.W_k(keys)                             # [B, S, d_key]
        sigma_K = self._activation(K)                   # [B, S, d_key]

        # Flatten batch and sequence for aggregation
        sigma_K_flat = sigma_K.reshape(-1, self.d_key)  # [B*S, d_key]
        V_flat = values.reshape(-1, self.d_value)        # [B*S, d_value]

        # Incremental update: M += σ(K)^T @ V
        self.M.add_(sigma_K_flat.t() @ V_flat)
        self.z.add_(sigma_K_flat.sum(dim=0))

        # Track statistics
        num_vectors = sigma_K_flat.shape[0]
        self._update_count.add_(1)
        self._total_vectors_stored.add_(num_vectors)

    def gate(self, A_mem: torch.Tensor, A_local: torch.Tensor) -> torch.Tensor:
        """Learnable gating between memory attention and local attention.

        Args:
            A_mem: Memory-retrieved attention output.
            A_local: Standard local attention output.

        Returns:
            Gated combination of memory and local attention.
        """
        g = torch.sigmoid(self.beta)
        return g * A_mem + (1.0 - g) * A_local

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        local_attention_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: retrieve from memory, optionally gate with local attention,
        then update memory with new KV pairs.

        Args:
            queries: [batch, seq_len, d_value]
            keys: [batch, seq_len, d_value]
            values: [batch, seq_len, d_value]
            local_attention_output: Optional local attention result for gating.

        Returns:
            Output tensor of shape [batch, seq_len, d_value].
        """
        # 1. Retrieve from compressive memory
        A_mem = self.retrieve(queries)

        # 2. Gate with local attention if provided
        if local_attention_output is not None:
            output = self.gate(A_mem, local_attention_output)
        else:
            output = A_mem

        # 3. Update memory with current segment's KV pairs
        self.update(keys, values)

        return output

    def reset_memory(self) -> None:
        """Clear compressive memory (e.g., between sessions)."""
        self.M.zero_()
        self.z.fill_(1e-8)
        self._update_count.zero_()
        self._total_vectors_stored.zero_()

    def get_stats(self) -> dict:
        """Return memory utilization statistics."""
        memory_norm = float(self.M.norm().item())
        return {
            "d_key": self.d_key,
            "d_value": self.d_value,
            "update_count": int(self._update_count.item()),
            "total_vectors_stored": int(self._total_vectors_stored.item()),
            "memory_matrix_norm": memory_norm,
            "gate_value": float(torch.sigmoid(self.beta).item()),
        }
