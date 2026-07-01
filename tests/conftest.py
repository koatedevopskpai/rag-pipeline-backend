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
