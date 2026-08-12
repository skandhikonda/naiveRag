"""FastAPI routes for the Naive RAG application."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SourceSchema(BaseModel):
    chunk_id: str
    chunk_number: int
    page_number: int
    distance: float


class RagResponseSchema(BaseModel):
    answer: str
    sources: List[SourceSchema]


app = FastAPI(title="Naive RAG API")

# Initialize shared services (lightweight clients only)
embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()


@app.post("/ask", response_model=RagResponseSchema)
def ask(request: AskRequest):
    """Accept a question and return a grounded answer with sources."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Create retrieval service per-request to allow custom top_k
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        top_k=request.top_k or 3,
    )

    rag_service = RAGService(retrieval_service=retrieval_service, embedding_service=embedding_service)

    try:
        rag_response = rag_service.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceSchema(
            chunk_id=src.chunk_id,
            chunk_number=src.chunk_number,
            page_number=src.page_number,
            distance=src.distance,
        )
        for src in rag_response.sources
    ]

    return RagResponseSchema(answer=rag_response.answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
def ui_index():
        """Simple HTML UI to ask questions and display RAG answers."""
        html = """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Naive RAG - Ask</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 2rem; }
                    textarea { width: 100%; height: 120px; }
                    pre { background: #f6f8fa; padding: 1rem; }
                    .sources { margin-top: 1rem; }
                </style>
            </head>
            <body>
                <h1>Naive RAG — Ask a Question</h1>
                <label for="question">Question</label>
                <textarea id="question" placeholder="Enter your question"></textarea>
                <br/>
                <label for="top_k">Top K</label>
                <input id="top_k" type="number" value="3" min="1" max="10" />
                <br/><br/>
                <button id="ask">Ask</button>

                <h2>Answer</h2>
                <pre id="answer">(no answer yet)</pre>

                <h3>Sources</h3>
                <div id="sources" class="sources">(no sources yet)</div>

                <script>
                    const askBtn = document.getElementById('ask');
                    askBtn.addEventListener('click', async () => {
                        const question = document.getElementById('question').value;
                        const top_k = parseInt(document.getElementById('top_k').value || '3');
                        document.getElementById('answer').textContent = 'Loading...';
                        document.getElementById('sources').textContent = '';

                        try {
                            const res = await fetch('/ask', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ question, top_k })
                            });

                            if (!res.ok) {
                                const err = await res.json();
                                document.getElementById('answer').textContent = 'Error: ' + (err.detail || res.statusText);
                                return;
                            }

                            const data = await res.json();
                            document.getElementById('answer').textContent = data.answer;

                            const srcDiv = document.getElementById('sources');
                            if (!data.sources || data.sources.length === 0) {
                                srcDiv.textContent = 'No sources.';
                            } else {
                                srcDiv.innerHTML = '<ul>' + data.sources.map(s => `<li>Page ${s.page_number}, Chunk ${s.chunk_number} (distance: ${s.distance})</li>`).join('') + '</ul>';
                            }
                        } catch (e) {
                            document.getElementById('answer').textContent = 'Request failed: ' + e;
                        }
                    });
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=200)
