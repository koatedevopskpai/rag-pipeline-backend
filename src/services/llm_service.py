# =============================================================================
# src/services/llm_service.py — LLM Service (Answer Generation)
# =============================================================================
# Context:
#   Wraps LangChain's ChatOpenAI to generate answers. The final step in
#   the RAG pipeline — retrieved context + user question → LLM → answer.
#
# Architectural Reasoning:
#   - ChatOpenAI from langchain-openai, not direct OpenAI SDK calls.
#     Same abstraction principle as the embedding service: swap providers
#     by changing one import and the model name in an env var.
#   - generate_with_context uses LangChain's ChatPromptTemplate + LCEL
#     (LangChain Expression Language) pipeline: prompt_template | llm.
#     This is the idiomatic LangChain way — composable, testable, and
#     trivially extensible (add a retriever step, add a parser step, etc.).
#   - astream_with_context yields chunks asynchronously for SSE streaming.
#     This is the difference between a toy RAG and a production RAG:
#     users won't wait 5-10 seconds for a full response. Streaming is
#     table stakes.
#   - SYSTEM_PROMPT is a class-level constant enforcing grounded answers.
#     "If the context does not contain enough information... say so."
#     This is the most important prompt engineering decision in the
#     entire system. Without it, the LLM will hallucinate answers from
#     its training data, defeating the purpose of RAG.
#
# Trade-offs:
#   - ChatPromptTemplate is recreated on every generate_with_context call.
#     In production, you'd cache the template. For this portfolio, the
#     overhead is negligible (~0.01ms).
#   - No retry logic. If the LLM API returns a 429 (rate limit) or 500,
#     the request fails. In production, add tenacity retries with
#     exponential backoff.
#   - No fallback model. If gpt-4o-mini is down, the system is down.
#     In production, you'd chain a fallback to gpt-3.5-turbo or a
#     local model.
#
# Opinionated Stance:
#   The system prompt is the single most important line of code in any
#   RAG system. Get it wrong and your AI confidently lies. Get it right
#   and it says "I don't know" when it should. Mine says the latter.
#   This is non-negotiable for any system that claims to be grounded.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant answering questions based on provided context.
Answer the question using only the information in the context below.
If the context does not contain enough information to answer the question, say so clearly.
Do not make up information or use knowledge beyond the provided context."""


class LLMService:
    def __init__(self) -> None:
        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            kwargs["openai_api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["openai_api_base"] = settings.openai_base_url

        self._llm = ChatOpenAI(**kwargs)
        self._model_name = settings.openai_model
        logger.info("Initialized LLMService with model=%s", self._model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = self._llm.invoke(messages)
        return response.content

    def generate_with_context(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt_template | self._llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    async def async_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = await self._llm.ainvoke(messages)
        return response.content

    async def async_generate_with_context(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt_template | self._llm
        response = await chain.ainvoke({"context": context, "question": question})
        return response.content

    async def astream_with_context(
        self,
        question: str,
        context: str,
    ):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt_template | self._llm
        async for chunk in chain.astream({"context": context, "question": question}):
            yield chunk

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def llm(self):
        return self._llm
