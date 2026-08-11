"""RAG (Retrieval-Augmented Generation) orchestration service."""

from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAIError
from dotenv import load_dotenv

from app.models.rag_response import RagResponse, Source
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class RAGService:
    """Orchestrates retrieval and grounded answer generation using OpenAI.

    The service retrieves top-K chunks for a question using `RetrievalService`,
    builds a context, and asks the LLM to answer using only that context.
    """

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        completion_model: Optional[str] = None,
    ) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = self.embedding_service.client
        self.completion_model = completion_model or "gpt-3.5-turbo"
        self.logger = logger

    def _build_context(self, retrieved_chunks) -> str:
        parts: List[str] = []
        for idx, c in enumerate(retrieved_chunks, start=1):
            header = f"[Source {idx} | Page {c.page_number} | Chunk {c.chunk_number} | Distance: {c.distance}]"
            parts.append(header)
            parts.append(c.text)
        return "\n\n".join(parts)

    def answer(self, question: str) -> RagResponse:
        """Answer a user's question using retrieved document context.

        Args:
            question: Natural-language user question.

        Returns:
            RagResponse containing the generated answer and sources.
        """
        if not question or not question.strip():
            self.logger.error("Empty question provided to RAGService.")
            raise ValueError("Question must not be empty.")

        q = question.strip()
        self.logger.info("RAGService received question: %s", q)

        try:
            retrieved = self.retrieval_service.retrieve(q)
        except Exception as exc:
            self.logger.exception("Retrieval failed in RAGService.")
            raise RuntimeError("Failed to retrieve context for the question.") from exc

        self.logger.info("Number of chunks retrieved: %d", len(retrieved))

        if not retrieved:
            msg = "I could not find relevant information in the provided documents."
            self.logger.info("No retrieved chunks; returning fallback message.")
            return RagResponse(answer=msg, sources=[])

        context = self._build_context(retrieved)

        system_instructions = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "If the answer cannot be found in the context, say explicitly that the information is not available in the provided documents. "
            "Include source citations (page and chunk) when stating facts. Be concise."
        )

        user_prompt = f"Context:\n\n{context}\n\nQuestion: {q}\n\nAnswer:"

        try:
            self.logger.info("Sending completion request to OpenAI model: %s", self.completion_model)
            response = self.client.chat.completions.create(
                model=self.completion_model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            content = response.choices[0].message.content
            self.logger.info("Received completion response from OpenAI.")
        except OpenAIError as exc:
            self.logger.exception("OpenAI API error during RAG completion.")
            raise RuntimeError("OpenAI API error during RAG completion.") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error during RAG completion.")
            raise RuntimeError("Failed to generate an answer via OpenAI.") from exc

        sources: List[Source] = [
            Source(
                chunk_id=str(c.chunk_id),
                chunk_number=int(c.chunk_number),
                page_number=int(c.page_number),
                distance=float(c.distance),
            )
            for c in retrieved
        ]

        return RagResponse(answer=content.strip(), sources=sources)
