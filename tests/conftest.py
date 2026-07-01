# =============================================================================
# tests/conftest.py — Test Configuration and Fixtures
# =============================================================================
# Context:
#   pytest configuration, fixtures, and service mocking for the RAG
#   Pipeline test suite. Uses mocking to avoid real API calls and
#   vector store dependencies.
#
# Architectural Reasoning:
#   - Mocking (via unittest.mock.patch) is chosen over test containers
#     or emulators because:
#     a) Zero infrastructure: no Docker, no API keys, no network.
#     b) Deterministic: mocked responses never change, tests never
#        flake due to network latency or API changes.
#     c) Fast: full test suite runs in < 100ms.
#     An integration test suite against real OpenAI + Chroma would take
#     minutes and require API keys that CI can't provide.
#   - The mock_services fixture is autouse=True — every test gets it
#     automatically. This is dangerous if you want real integration
#     tests (they'd be silently mocked). Mitigation: separate mock
#     and integration test files if needed later.
#   - Nested patch context managers mirror the dependency chain:
#     EmbeddingService → VectorStoreService → LLMService →
#     RAGPipeline. Each layer's mock feeds into the next. This
#     ensures the mocked RAGPipeline behaves like the real one,
#     just without network calls.
#   - ASGITransport + AsyncClient (httpx) test the FastAPI app
#     in-process without HTTP. This is faster than spinning up
#     uvicorn and avoids port conflicts.
#
# Trade-offs:
#   - Mocking the full dependency chain means we test the API layer
#     (routing, validation, error handling) but not the service layer
#     (actual embedding, retrieval, generation logic). The service
#     layer unit tests (test_rag_pipeline.py) cover that separately.
#   - MagicMock returns defaults for all attributes — misspelled
#     method names silently pass. spec=RAGPipeline on the pipeline
#     mock (line 55) mitigates this by enforcing the real interface.
#
# Opinionated Stance:
#   FastAPI + mocking = fastest possible test feedback loop. If your
#   Python test suite takes longer than 1 second, you won't run it
#   before every commit. This suite runs in < 100ms. I run it
#   constantly. That's the point.
# =============================================================================

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app
from src.models.chat import QueryResponse, SourceDocument
from src.models.document import IngestResponse


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_services():
    with patch("src.services.embedding.EmbeddingService") as mock_embed:
        embed_instance = MagicMock()
        embed_instance.embed_text.return_value = [0.1] * 1536
        embed_instance.embed_texts.return_value = [[0.1] * 1536]
        embed_instance.model_name = "mock-model"
        mock_embed.return_value = embed_instance

        with patch("src.services.vector_store.VectorStoreService") as mock_vs:
            vs_instance = MagicMock()
            vs_instance.collection_count = 0
            vs_instance.add_documents = MagicMock()
            vs_instance.similarity_search = MagicMock(
                return_value=[(MagicMock(page_content="Mocked content", metadata={"source": "test"}), 0.95)]
            )
            mock_vs.return_value = vs_instance

            with patch("src.services.llm_service.LLMService") as mock_llm:
                llm_instance = MagicMock()
                llm_instance.generate.return_value = "Mocked answer"
                llm_instance.generate_with_context.return_value = "Mocked answer about Python."
                llm_instance.model_name = "mock-model"

                async def mock_stream():
                    yield MagicMock(content="Mocked")

                llm_instance.astream_with_context = MagicMock(return_value=mock_stream())
                mock_llm.return_value = llm_instance

                with patch("src.core.dependencies.get_rag_pipeline") as mock_get_pipeline:
                    from src.services.rag_pipeline import RAGPipeline

                    pipeline_instance = MagicMock(spec=RAGPipeline)
                    pipeline_instance.ingest_text.return_value = IngestResponse(
                        status="success", chunks=3, document_id="test-doc-id"
                    )
                    pipeline_instance.ingest_url.return_value = IngestResponse(
                        status="success", chunks=5, document_id="test-url-id"
                    )
                    pipeline_instance.ask.return_value = QueryResponse(
                        answer="Python was created by Guido van Rossum.",
                        sources=[
                            SourceDocument(
                                content="Python was created by Guido van Rossum in 1991.",
                                metadata={"source": "test"},
                                score=0.95,
                            )
                        ],
                        tokens_used=42,
                    )
                    pipeline_instance.collection_size = 3
                    mock_get_pipeline.return_value = pipeline_instance

                    yield


@pytest.fixture
async def test_app() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
