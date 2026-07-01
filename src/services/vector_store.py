# =============================================================================
# src/services/vector_store.py — Chroma Vector Store Service
# =============================================================================
# Context:
#   Wraps LangChain's Chroma integration for document storage and semantic
#   retrieval. Chroma is an open-source embedding database — the brain of
#   the RAG system where all ingested document chunks live.
#
# Architectural Reasoning:
#   - Chroma over Pinecone/Qdrant/Weaviate: zero infrastructure. Chroma
#     runs as an embedded (in-process) database. No Docker, no cloud
#     account, no API key. The reviewer clones the repo and it just works.
#     For an interview portfolio, this is the killer feature.
#   - persist() is called after every add_documents() call. Chroma's
#     auto-persist flag is unreliable across versions. Explicit persist
#     ensures data survives a restart. The try/except handles the fact
#     that newer Chroma versions deprecated the persist() method
#     (in-memory mode doesn't need it).
#   - similarity_search_with_relevance_scores returns (Document, score)
#     tuples rather than plain Documents. The score is essential for
#     source attribution (see chat.py model). Without it, you can't
#     distinguish a 0.95-relevant result from a 0.05-irrelevant one.
#
# Trade-offs:
#   - In-process Chroma means the vector store doesn't survive container
#     restarts unless persist_directory is on a volume mount. The
#     docker-compose.yml handles this with a named volume.
#   - Chroma's relevance scores are cosine similarity (0-1 range), but
#     the raw values vary by embedding model. There's no universal
#     "0.7 means relevant" threshold — you must tune per deployment.
#   - delete_collection() is destructive but useful for demos. In
#     production, you'd version collections rather than delete them.
#   - collection_count accesses a private attribute (_collection).
#     Chroma's public API doesn't expose count(). This is fragile —
#     a Chroma upgrade could break it. The try/except catches that.
#
# Opinionated Stance:
#   For interview portfolios, start with Chroma. It removes the
#   infrastructure barrier that stops reviewers from running your code.
#   When you go to production, swap to Qdrant or Pinecone for scale.
#   The LangChain abstraction makes this a one-line change.
# =============================================================================

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
