from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    upstream_reachable: bool = False
    vq_engine_loaded: bool = False
    device: str = "cpu"
