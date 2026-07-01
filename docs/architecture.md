# RAG Pipeline Architecture

## Why RAG?

Retrieval-Augmented Generation (RAG) addresses two fundamental limitations of large language models:

1. **Knowledge Cutoff**: LLMs are trained on data up to a certain date. RAG provides fresh, up-to-date context.
2. **Hallucination**: Without grounding, LLMs can fabricate information. RAG anchors answers in retrieved documents.
3. **Private Data**: Company-internal knowledge never needs to be in the training set — it stays in your vector database.

## Pipeline Steps

### Ingestion Pipeline (Write Path)

```
Document Upload ──► Text Splitter ──► Embeddings ──► Chroma DB
```

1. **Load**: Raw text or URL content enters the system
2. **Split**: `RecursiveCharacterTextSplitter` breaks documents into chunks (configurable size/overlap)
3. **Embed**: `OpenAIEmbeddings` converts each chunk into a vector
4. **Store**: Vectors + metadata stored in Chroma DB for similarity search

### Query Pipeline (Read Path)

```
User Query ──► Retriever ──► Context + Prompt ──► LLM ──► Answer
```

1. **Query**: User submits a question
2. **Retrieve**: Top-k most similar chunks retrieved via cosine similarity search
3. **Augment**: Retrieved chunks assembled into a context prompt
4. **Generate**: LLM produces grounded answer using only the provided context

## Component Decisions

### Chroma DB (Vector Store)
- **Local**: No external database service needed
- **Free**: MIT-licensed, no usage costs
- **Fast**: Efficient HNSW index for approximate nearest neighbor search
- **Persistent**: Saves to disk, survives restarts

### LangChain
- **Industry Standard**: De facto framework for LLM applications
- **Modular**: Swap components without changing architecture
- **Chain support**: Clean abstractions for prompt templates and LLM calls

### OpenAI-Compatible API
- **Portable**: Same code works with OpenAI, Azure OpenAI, local LLMs via LiteLLM/vLLM
- **Future-proof**: Switch providers by changing `OPENAI_BASE_URL`
- **Standard**: Chat completions API is the industry norm

## Scaling Considerations

### Async Architecture
- FastAPI async handlers prevent blocking on I/O
- LLM calls can use `ainvoke`/`astream` for concurrent processing

### Connection Pooling
- httpx (used internally) supports connection reuse
- Embedding service initialized once, reused across requests

### Caching
- `@lru_cache` on dependency functions ensures singleton services
- In-memory document cache can be added for frequently accessed data

### Horizontal Scaling
- Stateless API servers (vector store is the only stateful component)
- Chroma on network-attached storage for multi-replica deployments
- Load balancer in front of multiple uvicorn workers

## Directory Layout

```
src/
  main.py               # FastAPI app entry point
  core/
    config.py           # Pydantic settings (env-based)
    dependencies.py     # DI container (lru_cache singletons)
  api/
    health.py           # GET /api/v1/health
    ingest.py           # POST /api/v1/ingest/{upload,text,url}
    query.py            # POST /api/v1/query/{ask,stream}
  models/
    document.py         # Ingest request/response models
    chat.py             # Query request/response models
  services/
    embedding.py        # Text-to-vector embedding service
    vector_store.py     # Chroma DB wrapper
    llm_service.py      # LLM generation service
    rag_pipeline.py     # Orchestrates ingestion + RAG query
```
