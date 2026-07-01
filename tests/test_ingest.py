from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_text(test_app: AsyncClient) -> None:
    response = await test_app.post(
        "/api/v1/ingest/text",
        json={"text": "FastAPI is a modern web framework for building APIs with Python.", "source": "test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks"] >= 1
    assert len(data["document_id"]) > 0


@pytest.mark.asyncio
async def test_ingest_empty_text(test_app: AsyncClient) -> None:
    response = await test_app.post(
        "/api/v1/ingest/text",
        json={"text": "", "source": "test"},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_large_text(test_app: AsyncClient) -> None:
    paragraph = "This is a test paragraph. " * 500
    response = await test_app.post(
        "/api/v1/ingest/text",
        json={"text": paragraph, "source": "test_large"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks"] > 1
