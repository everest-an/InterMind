from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from latentcore.models.openai_compat import ChatCompletionRequest
from latentcore.services.proxy_service import ProxyService

router = APIRouter(tags=["proxy"])


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    proxy_svc = ProxyService(
        http_client=request.app.state.http_client,
        compression_svc=getattr(request.app.state, "compression_svc", None),
        context_analyzer=getattr(request.app.state, "context_analyzer", None),
    )

    if body.stream:
        return StreamingResponse(
            proxy_svc.stream_proxy(body, request.headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await proxy_svc.forward_proxy(body, request.headers)
