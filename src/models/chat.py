from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    tokens_used: int
