from fastapi import APIRouter, Request

from latentcore import __version__
from latentcore.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    state = request.app.state

    upstream_reachable = False
    http_client = getattr(state, "http_client", None)
    if http_client is not None:
        try:
            resp = await http_client.get("/models", timeout=3.0)
            upstream_reachable = resp.status_code < 500
        except Exception:
            pass

    vq_loaded = getattr(state, "vq_engine", None) is not None
    device = str(getattr(state, "device", "cpu"))

    return HealthResponse(
        status="ok",
        version=__version__,
        upstream_reachable=upstream_reachable,
        vq_engine_loaded=vq_loaded,
        device=device,
    )
