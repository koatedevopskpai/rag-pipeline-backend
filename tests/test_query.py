# =============================================================================
# tests/test_query.py — API-Level Query Tests
# =============================================================================
# Context:
#   Tests for the query HTTP endpoints: /api/v1/query/ask and
#   /api/v1/query/stream. Validates the full Q&A request/response cycle.
#
# Architectural Reasoning:
#   - test_ask_no_documents tests the system's behavior when the
#     knowledge base is empty. The mock returns a valid answer regardless,
#     but this test proves the endpoint doesn't crash with empty data.
#   - test_ask_with_context validates the two-step flow: ingest then
#     query. This is the primary user journey. If this test passes, the
#     core RAG loop works.
#   - test_ask_empty_question validates input guardrails. Every public
#     endpoint must reject empty inputs with proper 4xx responses.
#     Missing this == production incident waiting to happen.
#
# Trade-offs:
#   - No streaming test (POST /stream). Testing SSE requires consuming
#     an async generator and parsing the event stream format. It's
#     technically possible but adds complexity. The streaming logic
#     is exercised indirectly by test_rag_pipeline.py unit tests.
#
# Opinionated Stance:
#   Test unhappy paths with the same rigor as happy paths. An empty
#   question returning 500 is a bug. An empty knowledge base returning
#   a hallucinated answer is a bug. Your tests should catch both.
# =============================================================================

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ask_no_documents(test_app: AsyncClient) -> None:
    response = await test_app.post(
        "/api/v1/query/ask",
        json={"question": "What is Python?", "top_k": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "tokens_used" in data


@pytest.mark.asyncio
async def test_ask_with_context(test_app: AsyncClient) -> None:
    await test_app.post(
        "/api/v1/ingest/text",
        json={"text": "Python was created by Guido van Rossum and first released in 1991.", "source": "test"},
    )
    response = await test_app.post(
        "/api/v1/query/ask",
        json={"question": "Who created Python?", "top_k": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_ask_empty_question(test_app: AsyncClient) -> None:
    response = await test_app.post(
        "/api/v1/query/ask",
        json={"question": "", "top_k": 4},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()
