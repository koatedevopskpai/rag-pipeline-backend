# =============================================================================
# tests/test_ingest.py — API-Level Ingestion Tests
# =============================================================================
# Context:
#   Tests for the ingestion HTTP endpoints: /api/v1/ingest/text and
#   /api/v1/ingest/upload. Validates routing, input validation, error
#   handling, and response formatting.
#
# Architectural Reasoning:
#   - AsyncClient (httpx) tests the full FastAPI request/response cycle
#     including serialization, validation, and error middleware. This
#     catches bugs that unit tests miss (wrong status code, missing
#     fields in response, incorrect error format).
#   - Three test cases cover the three states: success (normal text),
#     validation failure (empty text → 400), and edge case (large text →
#     multi-chunk). This is the minimal set that exercises all code paths.
#   - test_upload_text uses POST /api/v1/ingest/text (not /upload)
#     because the file upload test would require constructing a real
#     UploadFile, which is unnecessary complexity when the /text
#     endpoint exercises the same pipeline.
#
# Trade-offs:
#   - No file upload test. Testing UploadFile via AsyncClient requires
#     multipart form data construction, which adds complexity for
#     marginal coverage gain since /text exercises the same validation
#     and pipeline logic.
#   - Asserts check response shape (chunks >= 1) not exact values,
#     because mock returns are deterministic. This is intentional —
#     exact-value tests break when mocks change, shape tests don't.
#
# Opinionated Stance:
#   Test the API from the outside in. Mock the internals, but test the
#   full HTTP contract (status codes, headers, response bodies). A unit
#   test that calls a function directly doesn't prove your API works.
# =============================================================================

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
