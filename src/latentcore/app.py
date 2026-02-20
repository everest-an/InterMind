import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from latentcore import __version__
from latentcore.config import get_settings
from latentcore.engine.device import resolve_device
from latentcore.engine.text_decoder import TextDecoder
from latentcore.engine.text_encoder import TextEncoder
from latentcore.engine.infini_attention import InfiniAttentionMemory
from latentcore.engine.vq_codebook import VectorQuantizer
from latentcore.middleware.error_handler import ErrorHandlerMiddleware
from latentcore.middleware.token_counter import TokenCounterMiddleware
from latentcore.routes import codebook_stats, health, latent, proxy
from latentcore.services.compression_service import CompressionService
from latentcore.services.context_analyzer import ContextAnalyzer
from latentcore.storage.database import init_database
from latentcore.storage.latent_store import LatentStore

logger = logging.getLogger("latentcore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve compute device
    device = resolve_device(settings.device)
    app.state.device = device
    logger.info("Compute device: %s", device)

    # HTTP client for upstream LLM
    app.state.http_client = httpx.AsyncClient(
        base_url=settings.upstream_base_url,
        timeout=httpx.Timeout(settings.upstream_timeout_s),
    )

    # VQ Codebook engine
    app.state.vq_engine = VectorQuantizer(
        num_embeddings=settings.codebook_size,
        embedding_dim=settings.embedding_dim,
        commitment_cost=settings.commitment_cost,
        ema_decay=settings.ema_decay,
        device=device,
    )

    # Text encoder / decoder
    app.state.text_encoder = TextEncoder(
        embedding_dim=settings.embedding_dim,
        device=device,
    )
    app.state.text_decoder = TextDecoder(
        embedding_dim=settings.embedding_dim,
        device=device,
    )

    # Infini-attention compressive memory
    app.state.infini_engine = InfiniAttentionMemory(
        d_key=settings.d_key,
        d_value=settings.d_value,
        device=device,
    )

    # SQLite database
    app.state.db = await init_database(settings.db_path)
    store = LatentStore(app.state.db)

    # Compression service
    app.state.compression_svc = CompressionService(
        vq_engine=app.state.vq_engine,
        text_encoder=app.state.text_encoder,
        text_decoder=app.state.text_decoder,
        store=store,
        infini_engine=app.state.infini_engine,
    )

    # Context analyzer for automatic compression decisions
    app.state.context_analyzer = ContextAnalyzer(settings)

    logger.info(
        "LatentCore v%s started | upstream=%s | codebook=%d×%d | device=%s",
        __version__,
        settings.upstream_base_url,
        settings.codebook_size,
        settings.embedding_dim,
        device,
    )

    yield

    # Shutdown
    await app.state.http_client.aclose()
    await app.state.db.close()
    logger.info("LatentCore shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LatentCore",
        description="Token compression gateway based on LatentMAS, Infini-attention, and VQ-VAE",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(TokenCounterMiddleware)

    app.include_router(health.router)
    app.include_router(proxy.router)
    app.include_router(latent.router)
    app.include_router(codebook_stats.router)

    return app
