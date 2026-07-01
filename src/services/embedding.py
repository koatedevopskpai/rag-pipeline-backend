# =============================================================================
# src/services/embedding.py — Embedding Service (Vector Representation)
# =============================================================================
# Context:
#   Wraps LangChain's OpenAIEmbeddings to convert text → vector embeddings.
#   Embeddings are the foundation of semantic search — without them, the
#   vector store can't measure document similarity.
#
# Architectural Reasoning:
#   - OpenAIEmbeddings from langchain-openai, not direct OpenAI API calls.
#     LangChain provides a unified interface: swap to HuggingFace, Cohere,
#     or any other provider by changing one import and the model name.
#     The caller (RAGPipeline) doesn't know or care which embedding model
#     is running.
#   - Constructor accepts optional params from settings, including
#     openai_api_base for provider-agnostic deployment (Azure OpenAI,
#     local LLMs via LiteLLM proxy, etc.).
#   - embed_text (single query) vs embed_texts (batch documents) are
#     separated because LangChain's API distinguishes them: embed_query
#     and embed_documents. Batching is critical for ingestion performance.
#
# Trade-offs:
#   - No embedding caching. Every call to embed_text/embed_texts hits the
#     API. For a portfolio project this is fine. For production with
#     repeated queries, you'd cache query embeddings in Redis or LRU.
#   - No embedding dimension validation. Different models return different
#     vector sizes (1536 for text-embedding-3-small, 3072 for large).
#     Chroma handles this automatically, but if you swap models, existing
#     embeddings become incompatible and you must re-embed all documents.
#
# Opinionated Stance:
#   Never hardcode an embedding model name in your application code.
#   It should be an environment variable. The day you need to swap from
#   OpenAI to a local embedding model for cost reasons, you want a
#   one-line env var change, not a code refactor.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings

from src.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
        }
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            kwargs["openai_api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["openai_api_base"] = settings.openai_base_url
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self._model_name = settings.embedding_model
        logger.info(
            "Initialized EmbeddingService with model=%s",
            self._model_name,
        )

    def embed_text(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    @property
    def model_name(self) -> str:
        return self._model_name
