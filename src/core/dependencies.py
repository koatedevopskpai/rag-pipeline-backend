# =============================================================================
# src/core/dependencies.py — Dependency Injection via lru_cache Singletons
# =============================================================================
# Context:
#   Service wiring layer. Connects EmbeddingService, VectorStoreService,
#   LLMService, and RAGPipeline into a dependency graph, then exposes
#   factory functions for FastAPI's Depends().
#
# Architectural Reasoning:
#   - lru_cache(maxsize=1) creates a singleton per factory function.
#     This means one instance of each service for the app's lifetime.
#     Chroma DB connection pooling, embedding model loading, and LLM
#     client initialization are all expensive operations — you want
#     them once, not per-request.
#   - FastAPI's Depends() integrates with this trivially. In the route
#     handlers, `pipeline: RAGPipeline = Depends(get_rag_pipeline)` is
#     clean, testable, and obvious.
#   - The dependency chain is explicit: VectorStoreService needs
#     EmbeddingService, RAGPipeline needs all three. No hidden
#     imports, no magic.
#   - No async factory needed because service initialization is
#     synchronous (Chroma connects lazily, OpenAI clients are
#     thread-safe).
#
# Trade-offs:
#   - lru_cache singletons can't be easily reset between tests without
#     calling the cached function's cache_clear(). The test suite
#     works around this via mocking (see tests/conftest.py) rather than
#     real service instantiation.
#   - No lazy initialization — all services are created at first
#     Depends() call, not at app startup. For a portfolio project,
#     this is fine. For production with cold-start-sensitive services,
#     you'd use a startup event to pre-warm the cache.
#
# Opinionated Stance:
#   FastAPI Depends() with lru_cache is the cleanest DI pattern in
#   Python. It's explicit, testable, and zero-config. If you're using
#   global variables or module-level state for service wiring, you're
#   making testing harder than it needs to be.
# =============================================================================

from __future__ import annotations

from functools import lru_cache

from src.services.embedding import EmbeddingService
from src.services.llm_service import LLMService
from src.services.rag_pipeline import RAGPipeline
from src.services.vector_store import VectorStoreService


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService(embedding_service=get_embedding_service())


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
    )
