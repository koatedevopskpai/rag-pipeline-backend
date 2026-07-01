from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(
        self,
        embedding_service: Any | None = None,
    ) -> None:
        embedding_kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
        }
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            embedding_kwargs["openai_api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            embedding_kwargs["openai_api_base"] = settings.openai_base_url

        self._embedding_function = OpenAIEmbeddings(**embedding_kwargs)

        self._vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embedding_function,
            persist_directory=settings.vector_store_path,
        )
        logger.info(
            "Initialized VectorStoreService at path=%s collection=%s",
            settings.vector_store_path,
            settings.collection_name,
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        self._vector_store.add_documents(documents=documents, ids=ids)
        try:
            self._vector_store.persist()
        except (AttributeError, NotImplementedError):
            pass
        logger.info("Added %d documents to vector store", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        results = self._vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )
        logger.info(
            "Similarity search for '%s' returned %d results",
            query[:50],
            len(results),
        )
        return results

    def delete_collection(self) -> None:
        self._vector_store.delete_collection()
        logger.info("Deleted collection '%s'", settings.collection_name)

    @property
    def collection_count(self) -> int:
        try:
            return self._vector_store._collection.count()
        except (AttributeError, Exception):
            return 0
