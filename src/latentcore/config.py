from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LATENTCORE_", env_file=".env")

    # Upstream LLM
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = ""
    upstream_timeout_s: float = 120.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # VQ Codebook
    codebook_size: int = 8192
    embedding_dim: int = 768
    commitment_cost: float = 0.25
    ema_decay: float = 0.99

    # Infini-Attention
    d_key: int = 64
    d_value: int = 64
    segment_length: int = 2048

    # Context analysis
    compress_threshold_tokens: int = 1000
    max_input_tokens: int = 128000

    # Storage
    db_path: str = "latentcore.db"

    # Device
    device: str = "auto"

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
