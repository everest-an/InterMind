import torch
import pytest

from latentcore.engine.vq_codebook import VectorQuantizer


class TestVectorQuantizer:
    def test_encode_shape(self, vq_engine):
        z = torch.randn(2, 10, vq_engine.embedding_dim)
        quantized, indices = vq_engine.encode(z)

        assert quantized.shape == z.shape
        assert indices.shape == (2, 10)
        assert indices.dtype == torch.int64

    def test_encode_indices_in_range(self, vq_engine):
        z = torch.randn(1, 20, vq_engine.embedding_dim)
        _, indices = vq_engine.encode(z)

        assert indices.min() >= 0
        assert indices.max() < vq_engine.num_embeddings

    def test_decode_shape(self, vq_engine):
        indices = torch.randint(0, vq_engine.num_embeddings, (2, 10))
        decoded = vq_engine.decode(indices)

        assert decoded.shape == (2, 10, vq_engine.embedding_dim)

    def test_encode_decode_roundtrip(self, vq_engine):
        """Encoding then decoding should return codebook vectors."""
        z = torch.randn(1, 5, vq_engine.embedding_dim)
        quantized, indices = vq_engine.encode(z)
        decoded = vq_engine.decode(indices)

        torch.testing.assert_close(quantized, decoded)

    def test_forward_returns_loss(self, vq_engine):
        vq_engine.train()
        z = torch.randn(1, 10, vq_engine.embedding_dim, requires_grad=True)
        quantized_st, indices, loss = vq_engine(z)

        assert quantized_st.shape == z.shape
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_straight_through_gradient(self, vq_engine):
        """Gradients should flow through the straight-through estimator."""
        vq_engine.train()
        z = torch.randn(1, 5, vq_engine.embedding_dim, requires_grad=True)
        quantized_st, _, loss = vq_engine(z)

        total_loss = quantized_st.sum() + loss
        total_loss.backward()

        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_usage_tracking(self, vq_engine):
        z = torch.randn(1, 10, vq_engine.embedding_dim)
        vq_engine.encode(z)

        stats = vq_engine.get_stats()
        assert stats["total_encodes"] == 10
        assert stats["active_codes"] > 0
        assert stats["active_codes"] <= 10

    def test_reset_usage_stats(self, vq_engine):
        z = torch.randn(1, 10, vq_engine.embedding_dim)
        vq_engine.encode(z)
        vq_engine.reset_usage_stats()

        stats = vq_engine.get_stats()
        assert stats["total_encodes"] == 0
        assert stats["active_codes"] == 0

    def test_get_stats_structure(self, vq_engine):
        stats = vq_engine.get_stats()
        assert "codebook_size" in stats
        assert "embedding_dim" in stats
        assert "active_codes" in stats
        assert "usage_rate" in stats
        assert "total_encodes" in stats
