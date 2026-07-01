# =============================================================================
# tests/conftest.py — Test Configuration and Fixtures
# =============================================================================
# Context:
#   pytest configuration, fixtures, and service mocking for the RAG
#   Pipeline test suite. Mocks at the OpenAI SDK level so the real
#   service logic runs but no external API calls are made.
#
# Architectural Reasoning:
#   - Patching is done at the LangChain import level (OpenAIEmbeddings
#     and ChatOpenAI) rather than at the dependency injection level.
#     This means get_rag_pipeline() returns a REAL RAGPipeline with
#     REAL service instances, but all AI backends are mocked.
#   - MockChatOpenAI is a purpose-built class (not MagicMock) that
#     supports LangChain's LCEL | (pipe) operator. MagicMock doesn't
#     handle __ror__ properly for LCEL chain composition.
#   - Both embedding.py AND vector_store.py import OpenAIEmbeddings
#     separately — both import paths are patched.
#   - lru_cache is cleared after patching so the DI layer creates
#     fresh services that pick up the mocked constructors.
#   - ASGITransport + AsyncClient test the FastAPI app in-process.
#
# Trade-offs:
#   - Mocking at the constructor level means service init runs with
#     fake backends. If init had side effects that depend on API
#     responses, this would break. For this portfolio, all init is
#     synchronous and stateless.
#   - Hardcoded return values limit test scenarios. For deeper
#     testing, you'd swap the mock class per test case.
#
# Opinionated Stance:
#   Mock at the lowest integration point possible. For AI services,
#   that's the AI SDK client, not your service wrapper. Mocking
#   at the DI level means you're testing mocks, not your code.
# =============================================================================

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk

from src.main import create_app

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")


class MockChatOpenAI:
    """Drops-in replacement for langchain_openai.ChatOpenAI.

    LangChain's LCEL pipe (|) operator calls coerce_to_runnable()
    on the right-hand side. That function checks isinstance(thing, Runnable),
    callable(thing), or isinstance(thing, dict). __call__ makes this
    instance callable, so coerce_to_runnable wraps it in RunnableCallable.
    All invocation methods return predictably canned responses.
    """
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model", "mock-model")

    def invoke(self, *args, **kwargs):
        return AIMessage(content="Mocked LLM response")

    async def ainvoke(self, *args, **kwargs):
        return AIMessage(content="Mocked LLM response")

    async def astream(self, *args, **kwargs):
        yield AIMessageChunk(content="Mocked")

    __call__ = invoke


class MockOpenAIEmbeddings:
    """Drops-in replacement for langchain_openai.OpenAIEmbeddings."""
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "mock-model")

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_ai_backends():
    """Mock OpenAI constructors so no real API calls are made.

    Patches both import paths for OpenAIEmbeddings (embedding.py
    and vector_store.py each import it independently).
    """
    patcher_embed = patch("src.services.embedding.OpenAIEmbeddings", MockOpenAIEmbeddings)
    patcher_vs_embed = patch("src.services.vector_store.OpenAIEmbeddings", MockOpenAIEmbeddings)
    patcher_llm = patch("src.services.llm_service.ChatOpenAI", MockChatOpenAI)

    patcher_embed.start()
    patcher_vs_embed.start()
    patcher_llm.start()

    from src.core import dependencies as deps
    deps.get_embedding_service.cache_clear()
    deps.get_vector_store.cache_clear()
    deps.get_llm_service.cache_clear()
    deps.get_rag_pipeline.cache_clear()

    yield

    patcher_embed.stop()
    patcher_vs_embed.stop()
    patcher_llm.stop()


@pytest.fixture
async def test_app() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
