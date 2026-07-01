from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.health import router as health_router
from src.api.ingest import router as ingest_router
from src.api.query import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("RAG Pipeline Backend starting up...")
    logger.info("OpenAI-compatible API configured")
    yield
    logger.info("RAG Pipeline Backend shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline Backend",
        version="1.0.0",
        description="Production-grade RAG Pipeline Backend with FastAPI and LangChain",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)

    return app


app = create_app()
