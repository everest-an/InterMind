from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EncodeVQRequest(BaseModel):
    text: str
    session_id: str | None = None


class EncodeVQResponse(BaseModel):
    ref_id: str
    indices: list[int]
    num_vectors: int
    original_token_count: int
    compression_ratio: float


class DecodeVQRequest(BaseModel):
    ref_id: str | None = None
    indices: list[int] | None = None


class DecodeVQResponse(BaseModel):
    text: str
    num_vectors: int


class LatentChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict] = Field(default_factory=list)
    latent_refs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class CodebookStatsResponse(BaseModel):
    codebook_size: int
    embedding_dim: int
    active_codes: int
    usage_rate: float
    total_encodes: int
