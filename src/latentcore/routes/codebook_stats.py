from fastapi import APIRouter, HTTPException, Request

from latentcore.models.latent import CodebookStatsResponse

router = APIRouter(tags=["latent"])


@router.get("/v1/latent/codebook/stats", response_model=CodebookStatsResponse)
async def codebook_stats(request: Request):
    """Return VQ codebook utilization statistics."""
    vq_engine = getattr(request.app.state, "vq_engine", None)
    if vq_engine is None:
        raise HTTPException(status_code=503, detail="VQ engine not initialized")

    stats = vq_engine.get_stats()
    return CodebookStatsResponse(**stats)
