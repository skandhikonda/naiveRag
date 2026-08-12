# Naive RAG

A minimal, educational Retrieval-Augmented Generation (RAG) pipeline implemented in Python.

This repository demonstrates the core RAG components implemented incrementally:

- Document loader ([app/services/document_load_service.py](app/services/document_load_service.py))
- Chunking service ([app/services/chunking_service.py](app/services/chunking_service.py))
- Embedding service ([app/services/embedding_service.py](app/services/embedding_service.py))
- Vector store (ChromaDB) ([app/services/vector_store_service.py](app/services/vector_store_service.py))
- Retrieval service ([app/services/retrieval_service.py](app/services/retrieval_service.py))
- RAG orchestration ([app/services/rag_service.py](app/services/rag_service.py))

This is a teaching project — the implementation is intentionally simple and framework-free.

## Requirements

- Python 3.10+ (tested with 3.12)
- A Python virtual environment
- An OpenAI API key with access to embeddings and chat/completions
- The packages in `requirements.txt`

## Setup (Windows)

1. Clone the repository and change into the project folder:

```powershell
cd path\to\ai-agentic-applications\naiveRag
```

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key and optional model override:

```env
OPENAI_API_KEY=sk-...
# Optional: customize embedding model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# Optional: override completion/chat model used by RAGService
OPENAI_COMPLETION_MODEL=gpt-3.5-turbo
```

Note: the code reads `OPENAI_EMBEDDING_MODEL` by default; the RAG service uses the client from the embedding service. You can also pass a `completion_model` when creating `RAGService`.

## Data

Place the PDF(s) you want to ingest into the `documents/` directory. A sample PDF is expected at:

- `documents/naive_rag_sample_5_page.pdf`

ChromaDB persistent storage is created under `data/chroma` by default when you run the ingestion pipeline.

## Running the ingestion + RAG flow

The `main.py` script runs the end-to-end ingest pipeline (load → chunk → embed → store) and then prompts you for a question to run the RAG flow.

Run:

```powershell
python main.py
```

Typical run steps performed by `main.py`:

- Load PDF pages using [app/services/document_load_service.py](app/services/document_load_service.py)
- Chunk pages using [app/services/chunking_service.py](app/services/chunking_service.py)
- Generate embeddings using [app/services/embedding_service.py](app/services/embedding_service.py)
- Persist embeddings and metadata to ChromaDB via [app/services/vector_store_service.py](app/services/vector_store_service.py)
- Prompt for a question, retrieve top-K chunks with [app/services/retrieval_service.py](app/services/retrieval_service.py)
- Generate a grounded answer with [app/services/rag_service.py](app/services/rag_service.py)

The script prints the generated answer and lists the sources (page and chunk numbers).

## HTTP API (FastAPI)

A small HTTP API is available at `app.api.routes:app`.

To run the API locally (after installing dependencies):

```powershell
# from the project root
uvicorn app.api.routes:app --reload --host 0.0.0.0 --port 8000
```

Endpoint:

- `POST /ask` — Accepts JSON `{ "question": "...", "top_k": 3 }`, returns `{ "answer": "...", "sources": [...] }`.

Example using `curl`:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"What is retrieval-augmented generation?","top_k":3}'

Modify the C:\Windows\System32\drivers\etc\hosts file to add the user defined api endpoint.
http://mynaiverag.local:8000/
```



## Configuration

- Chunk size and overlap are configured when creating `ChunkingService` in `main.py` (defaults in the class can be changed).
- ChromaDB persistence directory defaults to `data/chroma` but can be overridden by passing a `persist_directory` to `VectorStoreService`.
- OpenAI models are controlled via environment variables or constructor arguments for services.

## Error handling and common issues

- Missing `OPENAI_API_KEY`: You will see a `ValueError` from the embedding service. Add it to `.env`.
- `ModuleNotFoundError: chromadb`: install dependencies with `pip install -r requirements.txt` and ensure your virtualenv is active.
- ChromaDB query `include` options depend on the installed `chromadb` version — the code uses `documents`, `metadatas`, and `distances`.

## Project structure

Key files:

- [main.py](main.py) — runner for ingestion + interactive RAG query
- [app/services/document_load_service.py](app/services/document_load_service.py)
- [app/services/chunking_service.py](app/services/chunking_service.py)
- [app/services/embedding_service.py](app/services/embedding_service.py)
- [app/services/vector_store_service.py](app/services/vector_store_service.py)
- [app/services/retrieval_service.py](app/services/retrieval_service.py)
- [app/services/rag_service.py](app/services/rag_service.py)
- [app/models/document_chunk.py](app/models/document_chunk.py)
- [app/models/embedded_document_chunk.py](app/models/embedded_document_chunk.py)
- [app/models/retrieved_chunk.py](app/models/retrieved_chunk.py)
- [app/models/rag_response.py](app/models/rag_response.py)

## Extending the project

- Add token-aware chunking or use libraries (e.g. tiktoken) for production workloads.
- Add a REST API (FastAPI) as a front-end to upload docs and ask questions.
- Add unit tests around each service.

## Notes for educators / learners

This repository is intentionally minimal to make the RAG concepts clear. It focuses on:

- Document loading
- Fixed-size chunking with overlap
- Embedding generation
- Local vector storage (ChromaDB)
- Nearest-neighbor retrieval
- Prompting the LLM with retrieved context and enforcing grounded answers

If you want me to add a `README` section with example output, CI/test steps, or a FastAPI wrapper, tell me which you'd prefer and I will add it.

---
Credits: adapted for learning from common RAG patterns.
