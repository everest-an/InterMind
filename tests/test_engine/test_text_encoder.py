import torch
import pytest

from latentcore.engine.text_encoder import TextEncoder


class TestTextEncoder:
    def test_encode_shape(self, text_encoder):
        result = text_encoder.encode("Hello world, this is a test.")
        assert result.dim() == 3  # [1, seq_len, embedding_dim]
        assert result.shape[0] == 1
        assert result.shape[2] == text_encoder.embedding_dim

    def test_encode_different_lengths(self, text_encoder):
        short = text_encoder.encode("Hi")
        long = text_encoder.encode("This is a much longer text with many more tokens in it.")
        assert long.shape[1] > short.shape[1]

    def test_token_count(self, text_encoder):
        count = text_encoder.token_count("Hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_text(self, text_encoder):
        # Empty string may produce 0 tokens or a special token
        count = text_encoder.token_count("")
        assert count == 0
