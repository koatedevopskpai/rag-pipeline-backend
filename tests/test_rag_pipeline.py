from __future__ import annotations

import pytest

from src.models.chat import SourceDocument
from src.models.document import IngestResponse
from src.services.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline() -> RAGPipeline:
    from src.core.dependencies import get_rag_pipeline
    return get_rag_pipeline()


def test_ingest_and_retrieve(pipeline: RAGPipeline) -> None:
    text = "Python is a high-level, interpreted programming language."
    ingest_response = pipeline.ingest_text(text=text, source="test_python")
    assert ingest_response.status == "success"
    assert ingest_response.chunks >= 1

    query_response = pipeline.ask(question="Who created Python?", top_k=4)
    assert len(query_response.answer) > 0
    assert isinstance(query_response.sources, list)
    assert isinstance(query_response.tokens_used, int)


def test_format_context(pipeline: RAGPipeline) -> None:
    text = "FastAPI is based on Starlette and Pydantic."
    pipeline.ingest_text(text=text, source="test_fastapi")

    text2 = "FastAPI supports dependency injection and automatic OpenAPI docs."
    pipeline.ingest_text(text=text2, source="test_fastapi2")

    query_response = pipeline.ask(question="What is FastAPI based on?", top_k=4)
    assert len(query_response.sources) > 0
    for source in query_response.sources:
        assert isinstance(source.content, str)
        assert isinstance(source.metadata, dict)
        assert isinstance(source.score, float)


def test_empty_question_raises(pipeline: RAGPipeline) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.ask(question="", top_k=4)


def test_empty_text_raises(pipeline: RAGPipeline) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.ingest_text(text="", source="test_empty")


def test_ingest_response_model() -> None:
    resp = IngestResponse(status="success", chunks=3, document_id="doc-123")
    assert resp.status == "success"
    assert resp.chunks == 3
    assert resp.document_id == "doc-123"


def test_source_document_model() -> None:
    src = SourceDocument(content="Test content", metadata={"source": "test"}, score=0.95)
    assert src.content == "Test content"
    assert src.metadata["source"] == "test"
    assert src.score == 0.95
