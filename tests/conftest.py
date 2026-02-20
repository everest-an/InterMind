import pytest
import torch

from latentcore.config import Settings
from latentcore.engine.device import resolve_device
from latentcore.engine.infini_attention import InfiniAttentionMemory
from latentcore.engine.text_decoder import TextDecoder
from latentcore.engine.text_encoder import TextEncoder
from latentcore.engine.vq_codebook import VectorQuantizer


@pytest.fixture
def settings():
    """Test settings with small model dimensions for fast tests."""
    return Settings(
        upstream_base_url="http://test-upstream:11434/v1",
        db_path=":memory:",
        codebook_size=256,
        embedding_dim=64,
        d_key=32,
        d_value=32,
        commitment_cost=0.25,
        ema_decay=0.99,
        device="cpu",
        compress_threshold_tokens=50,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def vq_engine(settings, device):
    return VectorQuantizer(
        num_embeddings=settings.codebook_size,
        embedding_dim=settings.embedding_dim,
        commitment_cost=settings.commitment_cost,
        ema_decay=settings.ema_decay,
        device=device,
    )


@pytest.fixture
def text_encoder(settings, device):
    return TextEncoder(embedding_dim=settings.embedding_dim, device=device)


@pytest.fixture
def text_decoder(settings, device):
    return TextDecoder(embedding_dim=settings.embedding_dim, device=device)


@pytest.fixture
def infini_engine(settings, device):
    return InfiniAttentionMemory(
        d_key=settings.d_key, d_value=settings.d_value, device=device
    )
