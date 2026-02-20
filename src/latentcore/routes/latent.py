from fastapi import APIRouter, HTTPException, Request

from latentcore.models.latent import (
    DecodeVQRequest,
    DecodeVQResponse,
    EncodeVQRequest,
    EncodeVQResponse,
)

router = APIRouter(prefix="/v1/latent", tags=["latent"])


@router.post("/encode_vq", response_model=EncodeVQResponse)
async def encode_vq(request: Request, body: EncodeVQRequest):
    """Encode text to VQ discrete indices."""
    compression_svc = getattr(request.app.state, "compression_svc", None)
    if compression_svc is None:
        raise HTTPException(status_code=503, detail="Compression service not initialized")

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = await compression_svc.compress(
        text=body.text,
        metadata={"session_id": body.session_id} if body.session_id else None,
    )

    return EncodeVQResponse(
        ref_id=result.ref_id,
        indices=result.indices,
        num_vectors=result.num_vectors,
        original_token_count=result.original_token_count,
        compression_ratio=result.compression_ratio,
    )


@router.post("/decode_vq", response_model=DecodeVQResponse)
async def decode_vq(request: Request, body: DecodeVQRequest):
    """Decode VQ indices back to text."""
    compression_svc = getattr(request.app.state, "compression_svc", None)
    if compression_svc is None:
        raise HTTPException(status_code=503, detail="Compression service not initialized")

    if body.ref_id is None and body.indices is None:
        raise HTTPException(status_code=400, detail="Either ref_id or indices must be provided")

    try:
        text = await compression_svc.decompress(ref_id=body.ref_id, indices=body.indices)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    num_vectors = len(body.indices) if body.indices else 0
    if body.ref_id and not body.indices:
        store = request.app.state.compression_svc.store
        ref_data = await store.load_ref(body.ref_id)
        num_vectors = ref_data["num_indices"] if ref_data else 0

    return DecodeVQResponse(text=text, num_vectors=num_vectors)


@router.post("/chat_completions")
async def latent_chat_completions(request: Request, body: dict):
    """Chat completions with mixed text + latent ref support.

    Accepts standard messages plus optional latent_refs for context injection.
    This endpoint resolves VQ refs and includes them as compressed context
    before forwarding to the upstream LLM.
    """
    compression_svc = getattr(request.app.state, "compression_svc", None)
    http_client = request.app.state.http_client

    latent_refs = body.pop("latent_refs", None)

    # If latent refs are provided, resolve and inject as system context
    if latent_refs and compression_svc:
        resolved_text_parts = []
        for ref_id in latent_refs:
            try:
                text = await compression_svc.decompress(ref_id=ref_id)
                resolved_text_parts.append(text)
            except ValueError:
                pass

        if resolved_text_parts:
            context_msg = {
                "role": "system",
                "content": (
                    "[LatentCore Restored Context]\n" + "\n---\n".join(resolved_text_parts)
                ),
            }
            messages = body.get("messages", [])
            # Insert after existing system message, or at the beginning
            insert_idx = 0
            if messages and messages[0].get("role") == "system":
                insert_idx = 1
            messages.insert(insert_idx, context_msg)
            body["messages"] = messages

    # Forward to upstream
    try:
        resp = await http_client.post("/chat/completions", json=body)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
