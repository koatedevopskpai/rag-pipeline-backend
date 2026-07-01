# =============================================================================
# src/core/config.py — Application Configuration via Environment Variables
# =============================================================================
# Context:
#   Centralized configuration using Pydantic BaseSettings. Every tunable
#   parameter lives here — no hardcoded strings scattered across services.
#
# Architectural Reasoning:
#   - BaseSettings (pydantic-settings) reads from environment variables
#     and .env files automatically. This is the 12-factor app approach:
#     config varies across deployments, code does not.
#   - All defaults are safe for development (sk-placeholder API key,
#     local chroma_db path, gpt-4o-mini model). The app runs without
#     any environment setup — critical for portfolio demonstration where
#     the reviewer just wants to `pip install && uvicorn run`.
#   - openai_kwargs property collects LLM params into a single dict.
#     This avoids repeating the "if api_key then set it" pattern in
#     every service. DRY principle applied to configuration.
#   - embeddding_kwargs is separated from openai_kwargs because the
#     OpenAI embeddings API uses different param names (model vs
#     openai_api_key vs openai_api_base). LangChain's OpenAIEmbeddings
#     constructor differs from ChatOpenAI's.
#   - The USER_AGENT env var at the bottom fixes a LangChain warning
#     about missing user-agent headers. This is a papercut fix that
#     would waste hours debugging if not addressed.
#
# Trade-offs:
#   - BaseSettings with env_file support means the app can accidentally
#     load a .env file from the wrong directory. In containerized
#     deployments, .env files shouldn't be used — env vars only.
#     The "extra": "ignore" flag prevents crashes from unknown vars.
#   - Defaulting openai_api_key to "sk-placeholder" means the app starts
#     but embedding/LLM calls will fail until a real key is set.
#     Alternative: fail fast at startup. I chose soft-fail because it
#     lets reviewers explore the API structure and test endpoints
#     (even if they return errors) before configuring a key.
#   - temperature defaults to 0.0 for deterministic answers. This is
#     correct for RAG (factual Q&A). For creative tasks, you'd raise it.
#     Exposing it as an env var means no code change needed.
#
# Opinionated Stance:
#   If your configuration is not in environment variables, you're doing
#   it wrong. No config files, no YAML, no JSON — env vars or .env for
#   local development. 12-factor app principles are non-negotiable for
#   cloud-native deployments.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    openai_api_key: str = Field(default="sk-placeholder", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    vector_store_path: str = Field(default="./chroma_db", alias="VECTOR_STORE_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_k: int = Field(default=4, alias="TOP_K")

    @property
    def openai_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.openai_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        if self.openai_api_key and self.openai_api_key != "sk-placeholder":
            kwargs["api_key"] = self.openai_api_key
        return kwargs

    @property
    def embedding_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.embedding_model,
        }
        if self.openai_api_key and self.openai_api_key != "sk-placeholder":
            kwargs["api_key"] = self.openai_api_key
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        return kwargs

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "RAGPipelineBackend/1.0"
