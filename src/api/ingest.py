from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette import status

from src.core.dependencies import get_rag_pipeline
from src.models.document import DocumentIngestRequest, IngestResponse, UrlIngestRequest
from src.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


@router.post("/upload", response_model=IngestResponse)
async def upload_file(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported",
        )
    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 text",
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content cannot be empty",
        )

    try:
        result = pipeline.ingest_text(text=text, source=file.filename)
        logger.info("Uploaded file '%s' ingested successfully", file.filename)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    request: DocumentIngestRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> IngestResponse:
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty",
        )
    try:
        result = pipeline.ingest_text(text=request.text, source=request.source)
        logger.info("Text ingested successfully from source='%s'", request.source)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    request: UrlIngestRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> IngestResponse:
    try:
        result = pipeline.ingest_url(url=str(request.url))
        logger.info("URL '%s' ingested successfully", request.url)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
