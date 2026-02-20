from __future__ import annotations

import json
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger("latentcore.metrics")


class TokenCounterMiddleware(BaseHTTPMiddleware):
    """Adds X-Latentcore-* headers to responses with token counting metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Latentcore-Latency-Ms"] = f"{elapsed_ms:.1f}"

        # Add compression stats if available from the request context
        compression_stats = getattr(request.state, "compression_stats", None)
        if compression_stats:
            response.headers["X-Latentcore-Original-Tokens"] = str(
                compression_stats.get("original_tokens", 0)
            )
            response.headers["X-Latentcore-Compressed-Tokens"] = str(
                compression_stats.get("compressed_tokens", 0)
            )
            ratio = compression_stats.get("compression_ratio", 1.0)
            response.headers["X-Latentcore-Compression-Ratio"] = f"{ratio:.2f}"

        return response
