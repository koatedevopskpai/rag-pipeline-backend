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
