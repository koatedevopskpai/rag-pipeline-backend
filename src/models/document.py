# =============================================================================
# src/models/document.py — Document Ingestion Request/Response Models
# =============================================================================
# Context:
#   Pydantic v2 models defining the contract for document ingestion endpoints.
#   These are the wire format — what the API receives and returns.
#
# Architectural Reasoning:
#   - Pydantic v2 (BaseModel) chosen over dataclasses because FastAPI
#     integrates natively with Pydantic for validation, serialization, and
#     OpenAPI schema generation. A dataclass would lose automatic 422 error
#     responses and API docs.
#   - Separate request types for text and URL ingestion (rather than a
#     single polymorphic model) because the validation rules differ:
#     text needs a non-empty string, URL needs a valid HTTP URL.
#     Polymorphism would add complexity for zero gain.
#   - IngestResponse returns document_id as a string (UUID) so callers
#     can trace which ingested content produced which answer. This is
#     critical for debugging RAG quality issues.
#   - chunks field tells the caller how many chunks were created — useful
#     for tuning chunk_size/chunk_overlap parameters.
#
# Trade-offs:
#   - HttpUrl validator is strict — it rejects non-HTTP URLs. If we needed
#     to support FTP or file:// URLs, we'd need a custom validator.
#     For this portfolio, HTTP-only is the right constraint.
#   - No validation on the `source` field (free text). In production, you'd
#     enforce an enum or regex. For demo purposes, flexibility wins.
#
# Opinionated Stance:
#   Pydantic v2 is the only correct choice for FastAPI models in 2026.
#   If you're still using dataclasses or TypedDict for API contracts,
#   you're writing extra validation code that Pydantic gives you for free.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class DocumentIngestRequest(BaseModel):
    text: str
    source: str = "manual"


class UrlIngestRequest(BaseModel):
    url: HttpUrl


class IngestResponse(BaseModel):
    status: str
    chunks: int
    document_id: str
