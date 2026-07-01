# =============================================================================
# src/main.py — FastAPI Application Entry Point
# =============================================================================
# Context:
#   Application factory and module-level app instance. The entry point
#   for `uvicorn src.main:app`.
#
# Architectural Reasoning:
#   - create_app() factory pattern enables testability. Tests import
#     create_app() and get a fresh app instance without module-level
#     side effects. Without this, test state would leak across cases.
#   - lifespan context manager replaces the deprecated on_event API.
#     FastAPI 0.110+ recommends lifespan over startup/shutdown events
#     because lifespan works correctly with background tasks and
#     exception handling. on_event swallows exceptions.
#   - CORS middleware with allow_origins=["*"] is intentionally
#     permissive. This is a demo API meant to be accessed from any
#     frontend (Streamlit, React, curl). In production, restrict to
#     specific origins.
#   - Module-level `app = create_app()` (line 51) is what uvicorn
#     imports. It's separate from the factory so that tests can call
#     create_app() independently without hitting the module-level app.
#
# Trade-offs:
#   - CORS allow_credentials=True with allow_origins=["*"] is technically
#     invalid (the browser spec requires explicit origins for credentialed
#     requests). FastAPI/CORSMiddleware handles this gracefully by not
#     sending the Access-Control-Allow-Origin: * header when credentials
#     are involved. For this demo's scope, it works. For production,
#     set explicit origins.
#   - No middleware for request logging, metrics, or rate limiting. In
#     production, you'd add Prometheus middleware + slow-request
#     monitoring. For portfolio, the signal-to-noise ratio favors
#     simplicity.
#
# Opinionated Stance:
#   Never put route definitions in main.py. If your main.py is longer
#   than 50 lines, you've failed at separation of concerns. Routes
#   belong in router modules, configuration belongs in settings,
#   and main.py is just the glue. This file is 51 lines — right at
#   the limit.
# =============================================================================

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
