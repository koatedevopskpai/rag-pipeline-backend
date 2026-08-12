# RAG Pipeline Backend

![CI](https://github.com/koate-kpai/rag-pipeline-backend/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1.0%2B-1C3C3C?logo=langchain)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade **Retrieval-Augmented Generation (RAG)** backend built with FastAPI, LangChain, Chroma DB, and OpenAI-compatible APIs. Ingests documents, generates embeddings, stores them in a vector database, and answers questions using an LLM grounded in retrieved context.

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              Ingestion Pipeline              │
                     │                                              │
  Document Upload ──►│  Text Splitter  ──►  Embeddings  ──►  DB    │
                     │  (RecursiveChar)   (OpenAI Ada)   (Chroma)  │
                     └──────────────────────────────────────────────┘

                     ┌─────────────────────────────────────────────┐
                     │               Query Pipeline                 │
                     │                                              │
  User Question ──►│  Retriever  ──►  Context + Prompt  ──►  LLM  │
                     │  (Top-K)          (Augment)        (Answer) │
                     └──────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **API Framework** | FastAPI (async) | High performance, automatic OpenAPI docs, Pydantic v2 |
| **Vector Store** | Chroma DB | Local, free, persistent, no external service needed |
| **LLM Orchestration** | LangChain | Industry standard, modular, composable |
| **Embeddings** | text-embedding-3-small | OpenAI's most cost-effective embedding model |
| **LLM** | gpt-4o-mini (or any OpenAI-compatible) | Fast, cheap, swappable via env var |
| **Validation** | Pydantic v2 | Type-safe request/response models |
| **Testing** | pytest + httpx | Async test client for full integration tests |

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key (or any OpenAI-compatible endpoint)

### Installation

```bash
git clone https://github.com/koate-kpai/rag-pipeline-backend.git
cd rag-pipeline-backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install .
```

### Run

```bash
export OPENAI_API_KEY="sk-..."
uvicorn src.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern web framework for building APIs with Python.", "source": "docs"}'
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?", "top_k": 4}'
```

## Docker

```bash
docker compose up --build
```

Or build and run manually:

```bash
docker build -t rag-pipeline-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-..." rag-pipeline-backend
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/ingest/text` | Ingest raw text |
| POST | `/api/v1/ingest/upload` | Upload a `.txt` file |
| POST | `/api/v1/ingest/url` | Ingest content from a URL |
| POST | `/api/v1/query/ask` | Ask a question (returns answer + sources) |
| POST | `/api/v1/query/stream` | Ask a question (SSE streaming response) |

## Configuration

All configuration via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `sk-placeholder` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_BASE_URL` | `""` | For Azure OpenAI or local LLMs |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `VECTOR_STORE_PATH` | `./chroma_db` | Chroma persistence directory |
| `COLLECTION_NAME` | `documents` | Chroma collection name |
| `MAX_TOKENS` | `1024` | Max LLM response tokens |
| `TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |
| `TOP_K` | `4` | Default number of retrieved chunks |

## Project Structure

```
├── src/
│   ├── main.py               # FastAPI app entry point
│   ├── core/
│   │   ├── config.py         # Pydantic BaseSettings
│   │   └── dependencies.py   # Service DI (lru_cache singletons)
│   ├── api/
│   │   ├── health.py         # Health check endpoint
│   │   ├── ingest.py         # Document ingestion endpoints
│   │   └── query.py          # Query endpoints (ask + stream)
│   ├── models/
│   │   ├── document.py       # Ingest request/response models
│   │   └── chat.py           # Query request/response models
│   └── services/
│       ├── embedding.py      # OpenAI embedding service
│       ├── vector_store.py   # Chroma DB wrapper
│       ├── llm_service.py    # LLM generation service
│       └── rag_pipeline.py   # Orchestrates RAG pipeline
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── test_ingest.py        # Ingestion API tests
│   ├── test_query.py         # Query API tests
│   └── test_rag_pipeline.py  # Unit tests for pipeline
├── docs/
│   └── architecture.md       # Detailed architecture documentation
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Docker Compose setup
├── pyproject.toml            # Project metadata + dependencies
├── .github/workflows/ci.yml  # GitHub Actions CI
└── README.md
```

## Why This Matters for Python Developer / AI Roles

This project demonstrates:

- **Modern Python**: 3.11+ with type hints, async/await, Pydantic v2
- **Production Patterns**: Modular architecture, dependency injection, configuration management, logging
- **AI/LLM Engineering**: RAG implementation, vector embeddings, prompt engineering, context grounding
- **API Design**: FastAPI with proper status codes, error handling, OpenAPI docs, streaming
- **Testability**: Unit tests, integration tests, async fixtures, service isolation
- **DevOps Ready**: Docker multi-stage builds, CI pipeline, configuration via environment

## License

MIT
