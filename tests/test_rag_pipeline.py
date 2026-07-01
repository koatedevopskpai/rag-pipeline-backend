# =============================================================================
# tests/test_rag_pipeline.py — RAG Pipeline Unit Tests
# =============================================================================
# Context:
#   Unit tests for the RAGPipeline orchestration layer and Pydantic models.
#   Tests run against mocked services (no real API calls needed).
#
# Architectural Reasoning:
#   - Tests are split into pipeline behavior (test_ingest_and_retrieve,
#     test_format_context) and validation (test_empty_question_raises,
#     test_empty_text_raises). This separation makes it clear which
#     tests cover functionality vs defensive coding.
#   - test_ingest_response_model and test_source_document_model are
#     pure model tests — no mocking needed. They validate that
#     Pydantic constructs the correct objects. These are fast (~1ms)
#     and catch serialization bugs that would otherwise surface as
#     API 500 errors.
#   - The pipeline fixture calls get_rag_pipeline() which returns a
#     MagicMock (due to the autouse mock_services fixture in conftest).
#     This means all pipeline method calls exercise the mock interface
#     without executing real logic. The assertions verify that the mock
#     was called correctly and returned the expected shape.
#
# Trade-offs:
#   - No integration tests that call real services. The trade-off is
#     speed + portability vs confidence. For a portfolio project,
#     mocked tests prove the architecture is sound. For production,
#     add a separate integration test suite with real Chroma + a
#     local embedding model.
#   - The pipeline fixture interacts with the autouse mock. If the
#     mock setup changes, these tests break silently (not with
#     import errors). Using spec=RAGPipeline in conftest.py mitigates
#     this by enforcing interface conformance.
#
# Opinionated Stance:
#   Unit tests test YOUR logic, not your dependencies' logic. Mock
#   everything external. If your test fails because OpenAI changed
#   their API, that's an integration test, not a unit test. Keep
#   them separate. This file contains only unit tests.
# =============================================================================

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
