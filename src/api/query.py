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
