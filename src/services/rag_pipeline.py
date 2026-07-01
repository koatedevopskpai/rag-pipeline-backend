from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from src.core.config import settings
from src.models.chat import QueryResponse, SourceDocument
from src.models.document import IngestResponse
from src.services.embedding import EmbeddingService
from src.services.llm_service import LLMService
from src.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        llm_service: LLMService,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        logger.info("Initialized RAGPipeline")

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
    ) -> IngestResponse:
        if not text or not text.strip():
            raise ValueError("Text content cannot be empty")

        doc_id = str(uuid.uuid4())
        doc = Document(page_content=text, metadata={"source": source, "document_id": doc_id})
        chunks = self._text_splitter.split_documents([doc])

        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(len(chunks))]
        self._vector_store.add_documents(chunks, ids=chunk_ids)

        logger.info("Ingested %d chunks from source='%s'", len(chunks), source)
        return IngestResponse(
            status="success",
            chunks=len(chunks),
            document_id=doc_id,
        )

    def ingest_url(self, url: str) -> IngestResponse:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            raise ValueError(f"No content loaded from URL: {url}")

        combined_text = "\n\n".join(doc.page_content for doc in docs)
        return self.ingest_text(text=combined_text, source=url)

    def ask(self, question: str, top_k: int = 4) -> QueryResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        results = self._vector_store.similarity_search(query=question, k=top_k)

        sources: list[SourceDocument] = []
        context_parts: list[str] = []

        for doc, score in results:
            sources.append(
                SourceDocument(
                    content=doc.page_content[:500],
                    metadata=dict(doc.metadata) if doc.metadata else {},
                    score=float(score),
                )
            )
            context_parts.append(doc.page_content)

        context = "\n\n".join(context_parts)

        if not context.strip():
            answer = "No relevant documents found in the knowledge base to answer your question."
        else:
            answer = self._llm_service.generate_with_context(
                question=question,
                context=context,
            )

        tokens_used = self._estimate_tokens(context, question, answer)

        return QueryResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
        )

    async def ask_stream(self, question: str, top_k: int = 4) -> AsyncIterator[str]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        results = self._vector_store.similarity_search(query=question, k=top_k)

        sources: list[SourceDocument] = []
        context_parts: list[str] = []

        for doc, score in results:
            sources.append(
                SourceDocument(
                    content=doc.page_content[:500],
                    metadata=dict(doc.metadata) if doc.metadata else {},
                    score=float(score),
                )
            )
            context_parts.append(doc.page_content)

        context = "\n\n".join(context_parts)

        if not context.strip():
            yield f"data: {QueryResponse(answer='No relevant documents found.', sources=[], tokens_used=0).model_dump_json()}\n\n"
            return

        import json

        async for chunk in self._llm_service.astream_with_context(
            question=question,
            context=context,
        ):
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"

        yield f"data: {json.dumps({'sources': [s.model_dump() for s in sources]})}\n\n"
        yield "data: [DONE]\n\n"

    def _estimate_tokens(self, context: str, question: str, answer: str) -> int:
        return len(context.split()) + len(question.split()) + len(answer.split())

    @property
    def collection_size(self) -> int:
        return self._vector_store.collection_count
