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
