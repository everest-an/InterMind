from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from fastapi.responses import JSONResponse

from latentcore.models.openai_compat import ChatCompletionRequest, ChatMessage
from latentcore.services.compression_service import CompressionService
from latentcore.services.context_analyzer import ContextAnalyzer
from latentcore.utils.sse import format_sse_done, relay_sse_stream

logger = logging.getLogger("latentcore.proxy")


class ProxyService:
    """Proxies OpenAI-compatible requests to the upstream LLM.

    When context_analyzer and compression_svc are provided, automatically
    identifies and compresses long context segments before forwarding.
    Falls back to pure pass-through on any compression failure.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        compression_svc: CompressionService | None = None,
        context_analyzer: ContextAnalyzer | None = None,
    ):
        self.http_client = http_client
        self.compression_svc = compression_svc
        self.context_analyzer = context_analyzer

    async def _compress_context(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, dict]:
        """Analyze and compress long context segments.

        Returns the modified request and compression statistics.
        """
        stats = {"original_tokens": 0, "compressed_tokens": 0, "compression_ratio": 1.0}

        if self.context_analyzer is None or self.compression_svc is None:
            return request, stats

        try:
            analysis = self.context_analyzer.analyze(request.messages)

            if not analysis.segments_to_compress:
                return request, stats

            stats["original_tokens"] = analysis.total_input_tokens

            # Compress each identified segment
            compressed_summaries = []
            for segment in analysis.segments_to_compress:
                result = await self.compression_svc.compress(
                    text=segment.text,
                    metadata={"role": segment.role, "original_index": segment.original_index},
                )
                compressed_summaries.append(
                    ChatMessage(
                        role=segment.role,
                        content=(
                            f"[LatentCore compressed: {segment.token_count} tokens -> "
                            f"VQ ref {result.ref_id} ({result.num_vectors} indices, "
                            f"{result.compression_ratio:.1f}x compression)]"
                        ),
                    )
                )

            # Rebuild messages: kept messages + compressed summaries (in order)
            new_messages = []
            # Add system messages first
            for msg in analysis.messages_to_keep:
                if msg.role == "system":
                    new_messages.append(msg)

            # Add compressed summaries
            new_messages.extend(compressed_summaries)

            # Add non-system kept messages
            for msg in analysis.messages_to_keep:
                if msg.role != "system":
                    new_messages.append(msg)

            # Create modified request
            request_dict = request.model_dump(exclude_none=True)
            request_dict["messages"] = [m.model_dump(exclude_none=True) for m in new_messages]
            modified = ChatCompletionRequest(**request_dict)

            compressed_tokens = analysis.total_input_tokens - analysis.compressible_tokens
            # Add back the short summary tokens (rough estimate)
            compressed_tokens += len(compressed_summaries) * 30
            stats["compressed_tokens"] = compressed_tokens
            stats["compression_ratio"] = (
                analysis.total_input_tokens / compressed_tokens
                if compressed_tokens > 0
                else 1.0
            )

            logger.info(
                "Context compressed: %d -> ~%d tokens (%.1fx), %d segments compressed",
                analysis.total_input_tokens,
                compressed_tokens,
                stats["compression_ratio"],
                len(analysis.segments_to_compress),
            )

            return modified, stats

        except Exception:
            logger.exception("Context compression failed, falling back to pass-through")
            return request, stats

    def _build_upstream_payload(self, request: ChatCompletionRequest) -> dict:
        return request.model_dump(exclude_none=True)

    def _build_upstream_headers(self, original_headers) -> dict:
        headers = {"Content-Type": "application/json"}
        auth = original_headers.get("authorization")
        if auth:
            headers["Authorization"] = auth
        return headers

    async def forward_proxy(
        self, request: ChatCompletionRequest, original_headers
    ) -> JSONResponse:
        """Non-streaming proxy with automatic context compression."""
        request, stats = await self._compress_context(request)

        payload = self._build_upstream_payload(request)
        headers = self._build_upstream_headers(original_headers)

        try:
            resp = await self.http_client.post(
                "/chat/completions", json=payload, headers=headers
            )
            response = JSONResponse(status_code=resp.status_code, content=resp.json())

            # Add compression metrics headers
            if stats["compression_ratio"] > 1.0:
                response.headers["X-Latentcore-Original-Tokens"] = str(stats["original_tokens"])
                response.headers["X-Latentcore-Compressed-Tokens"] = str(stats["compressed_tokens"])
                response.headers["X-Latentcore-Compression-Ratio"] = (
                    f"{stats['compression_ratio']:.2f}"
                )

            return response
        except httpx.TimeoutException:
            logger.error("Upstream timeout")
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Upstream timeout", "type": "timeout", "code": 504}},
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to upstream")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "Cannot connect to upstream LLM",
                        "type": "connection_error",
                        "code": 502,
                    }
                },
            )

    async def stream_proxy(
        self, request: ChatCompletionRequest, original_headers
    ) -> AsyncIterator[str]:
        """Streaming proxy with automatic context compression."""
        request, stats = await self._compress_context(request)

        payload = self._build_upstream_payload(request)
        payload["stream"] = True
        headers = self._build_upstream_headers(original_headers)

        try:
            async with self.http_client.stream(
                "POST", "/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        error_data = json.loads(body)
                    except json.JSONDecodeError:
                        error_data = {
                            "error": {"message": body.decode(), "code": resp.status_code}
                        }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield format_sse_done()
                    return

                async for chunk in relay_sse_stream(resp):
                    yield chunk

        except httpx.TimeoutException:
            error = {"error": {"message": "Upstream timeout", "type": "timeout"}}
            yield f"data: {json.dumps(error)}\n\n"
            yield format_sse_done()
        except httpx.ConnectError:
            error = {
                "error": {"message": "Cannot connect to upstream", "type": "connection_error"}
            }
            yield f"data: {json.dumps(error)}\n\n"
            yield format_sse_done()
