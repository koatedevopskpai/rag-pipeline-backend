# =============================================================================
# src/api/query.py — RAG Query Endpoints (Ask + Stream)
# =============================================================================
# Context:
#   HTTP interface for asking questions against the ingested knowledge base.
#   Supports both synchronous (POST /ask) and streaming (POST /stream) modes.
#
# Architectural Reasoning:
#   - POST /ask returns a complete QueryResponse with answer + sources +
#     token count. This is the straightforward "give me the answer" path
#     that most clients use.
#   - POST /stream uses Server-Sent Events (SSE) for real-time token
#     delivery. SSE chosen over WebSocket because:
#     a) SSE is unidirectional (server → client), which is all we need.
#     b) SSE works over standard HTTP, no special proxy config needed.
#     c) SSE auto-reconnects in browsers.
#     WebSocket would be overkill for token streaming.
#   - The stream format is `data: {json}\n\n` (standard SSE). Each chunk
#     contains one token. The final messages include sources array and
#     [DONE] signal. This lets the client progressively render tokens
#     while waiting for the full source attribution.
#   - POST (not GET) because the request body contains a question string
#     that can exceed URL length limits. GET with query params would
#     truncate long questions and expose them in server logs.
#
# Trade-offs:
#   - SSE streams have no binary support. If we needed to stream images
#     or structured data alongside tokens, WebSocket would be required.
#   - No request ID or correlation ID in responses. For debugging across
#     distributed systems, you'd add a request_id header. For portfolio
#     scope, it's unnecessary complexity.
#
# Opinionated Stance:
#   If your RAG demo doesn't stream tokens, hiring managers will notice.
#   It's the single most visible UX differentiator between a toy and a
#   production system. SSE is trivial to implement (see below: ~20 lines)
#   and makes your demo look 10x more polished.
# =============================================================================

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from starlette.responses import StreamingResponse

from src.core.dependencies import get_rag_pipeline
from src.models.chat import QueryRequest, QueryResponse
from src.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["query"])


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )
    try:
        result = pipeline.ask(question=request.question, top_k=request.top_k)
        logger.info(
            "Question answered (top_k=%d) with %d sources",
            request.top_k,
            len(result.sources),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/stream")
async def ask_question_stream(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> StreamingResponse:
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    async def event_generator():
        async for chunk in pipeline.ask_stream(
            question=request.question,
            top_k=request.top_k,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
