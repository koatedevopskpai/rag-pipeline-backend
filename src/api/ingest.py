# =============================================================================
# src/api/ingest.py — Document Ingestion Endpoints
# =============================================================================
# Context:
#   HTTP interface for loading data into the knowledge base. Supports
#   three ingestion modes: raw text, file upload, and URL scraping.
#
# Architectural Reasoning:
#   - Three separate endpoints rather than one multipurpose endpoint
#     because validation logic differs per input type. Text needs
#     emptiness checks, files need extension + encoding validation,
#     URLs need reachability checks. Polymorphism would hide this.
#   - POST (not GET) for all ingestion endpoints because ingestion is
#     a state-changing operation. REST purists would argue for PUT
#     (idempotent create), but the system generates a new document_id
#     per request, making it non-idempotent. POST is correct.
#   - File upload restricted to .txt + UTF-8. This is intentionally
#     narrow. PDF, DOCX, and image support would require additional
#     parsers (PyMuPDF, python-docx, Tesseract). For portfolio demo,
#     .txt is sufficient. Adding file format support is a future
#     enhancement that makes a great interview talking point.
#   - ValueError is caught and converted to HTTP 400. Unhandled
#     exceptions become 500. The distinction matters: a bad input
#     (400) is the client's fault, a system crash (500) is ours.
#
# Trade-offs:
#   - WebBaseLoader in the URL endpoint is synchronous. For a long
#     page load, this blocks the async event loop. In production,
#     you'd run the loader in a thread pool executor or use an
#     async HTTP client directly.
#   - File upload max size is FastAPI's default (none). In production,
#     you'd enforce a 10MB limit. For demo, let it ride.
#
# Opinionated Stance:
#   Always distinguish client errors (4xx) from server errors (5xx).
#   A 500 for "empty text" is lazy and makes your API impossible to
#   debug. ValueError → 400, everything else → 500. Every endpoint
#   in this system follows that contract.
# =============================================================================

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
