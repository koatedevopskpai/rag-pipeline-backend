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
