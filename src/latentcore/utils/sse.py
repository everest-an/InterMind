from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def relay_sse_stream(response: httpx.Response) -> AsyncIterator[str]:
    """Relay SSE chunks from an upstream httpx streaming response."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                yield format_sse_done()
                return
            yield f"data: {payload}\n\n"
        elif line == "data:":
            continue


def format_sse_chunk(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_sse_done() -> str:
    return "data: [DONE]\n\n"
