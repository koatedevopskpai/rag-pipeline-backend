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
