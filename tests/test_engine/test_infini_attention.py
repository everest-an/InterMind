import torch
import pytest

from latentcore.engine.infini_attention import InfiniAttentionMemory


class TestInfiniAttentionMemory:
    def test_initial_memory_near_zero(self, infini_engine):
        assert infini_engine.M.abs().sum().item() < 1e-6

    def test_retrieve_from_empty_memory(self, infini_engine):
        queries = torch.randn(1, 5, infini_engine.d_value)
        result = infini_engine.retrieve(queries)

        assert result.shape == queries.shape
        # Should be near zero since memory is empty
        assert result.abs().max().item() < 1.0

    def test_update_changes_memory(self, infini_engine):
        keys = torch.randn(1, 5, infini_engine.d_value)
        values = torch.randn(1, 5, infini_engine.d_value)

        norm_before = infini_engine.M.norm().item()
        infini_engine.update(keys, values)
        norm_after = infini_engine.M.norm().item()

        assert norm_after > norm_before

    def test_retrieve_after_update_is_nonzero(self, infini_engine):
        keys = torch.randn(1, 10, infini_engine.d_value)
        values = torch.randn(1, 10, infini_engine.d_value)
        infini_engine.update(keys, values)

        queries = torch.randn(1, 3, infini_engine.d_value)
        result = infini_engine.retrieve(queries)

        assert result.shape == (1, 3, infini_engine.d_value)
        assert result.abs().sum().item() > 0

    def test_gate_interpolation(self, infini_engine):
        A_mem = torch.ones(1, 5, infini_engine.d_value)
        A_local = torch.zeros(1, 5, infini_engine.d_value)

        # With beta=0, sigmoid(0)=0.5, so output = 0.5 * A_mem + 0.5 * A_local
        output = infini_engine.gate(A_mem, A_local)
        expected = 0.5 * A_mem + 0.5 * A_local
        torch.testing.assert_close(output, expected)

    def test_forward_combines_retrieve_and_update(self, infini_engine):
        queries = torch.randn(1, 5, infini_engine.d_value)
        keys = torch.randn(1, 5, infini_engine.d_value)
        values = torch.randn(1, 5, infini_engine.d_value)

        output = infini_engine(queries, keys, values)
        assert output.shape == (1, 5, infini_engine.d_value)

        # Memory should have been updated
        stats = infini_engine.get_stats()
        assert stats["update_count"] == 1
        assert stats["total_vectors_stored"] == 5

    def test_reset_memory(self, infini_engine):
        keys = torch.randn(1, 10, infini_engine.d_value)
        values = torch.randn(1, 10, infini_engine.d_value)
        infini_engine.update(keys, values)

        infini_engine.reset_memory()

        assert infini_engine.M.abs().sum().item() < 1e-6
        stats = infini_engine.get_stats()
        assert stats["update_count"] == 0

    def test_get_stats(self, infini_engine):
        stats = infini_engine.get_stats()
        assert "d_key" in stats
        assert "d_value" in stats
        assert "update_count" in stats
        assert "total_vectors_stored" in stats
        assert "memory_matrix_norm" in stats
        assert "gate_value" in stats
        assert abs(stats["gate_value"] - 0.5) < 0.01  # sigmoid(0) ≈ 0.5

    def test_multiple_updates_accumulate(self, infini_engine):
        for _ in range(5):
            keys = torch.randn(1, 3, infini_engine.d_value)
            values = torch.randn(1, 3, infini_engine.d_value)
            infini_engine.update(keys, values)

        stats = infini_engine.get_stats()
        assert stats["update_count"] == 5
        assert stats["total_vectors_stored"] == 15
