# =============================================================================
# src/models/chat.py — Query/Response Models for RAG Q&A
# =============================================================================
# Context:
#   Pydantic v2 models for the RAG query/response contract. Defines how
#   questions are asked and how answers (with sources) are returned.
#
# Architectural Reasoning:
#   - QueryRequest.top_k is constrained to 1-20 via Field(ge=1, le=20).
#     This prevents clients from requesting 10,000 results and OOM-ing
#     the vector store. Defensive boundaries on every user input.
#   - SourceDocument bundles content, metadata, and a relevance score.
#     The score is critical — it lets the caller (or UI) show confidence.
#     Without it, users have no way to trust or doubt the answer.
#   - tokens_used in QueryResponse is an estimate (word count), not a
#     precise token count. This is intentional: exact token counting
#     requires a tokenizer that matches the LLM, which adds a dependency
#     (tiktoken) and coupling to the model provider. A word-count estimate
#     is good enough for portfolio demonstration.
#
# Trade-offs:
#   - SourceDocument.content is truncated to 500 chars in the pipeline
#     (see rag_pipeline.py). This prevents bloated responses but means
#     the UI can't show full context. For interview demo, 500 chars is
#     sufficient to show source attribution.
#   - metadata is untyped (dict). In production you'd use a TypedDict or
#     a BaseModel subclass. For flexibility across different document
#     loaders (PDF, web, database), untyped dict is pragmatic.
#
# Opinionated Stance:
#   Never return an LLM answer without source attribution. If you can't
#   show where the answer came from, you're not building RAG — you're
#   building a black box. SourceDocument.score is non-negotiable.
# =============================================================================

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
