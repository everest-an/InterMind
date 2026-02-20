from latentcore.config import Settings


def test_settings_defaults():
    s = Settings(upstream_base_url="http://localhost:11434/v1")
    assert s.codebook_size == 8192
    assert s.embedding_dim == 768
    assert s.port == 8000
    assert s.device == "auto"
    assert s.compress_threshold_tokens == 1000


def test_settings_override():
    s = Settings(
        upstream_base_url="http://custom:8080/v1",
        codebook_size=512,
        embedding_dim=128,
        device="cpu",
    )
    assert s.codebook_size == 512
    assert s.embedding_dim == 128
    assert s.device == "cpu"
    assert s.upstream_base_url == "http://custom:8080/v1"
