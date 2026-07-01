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
