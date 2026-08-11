"""Service for retrieving semantically similar document chunks from ChromaDB."""

from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAIError
from dotenv import load_dotenv

from app.models.retrieved_chunk import RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class RetrievalService:
    """Retrieve the most similar chunks for a user question."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        top_k: int = 3,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.top_k = top_k
        self.logger = logger

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Retrieve the top-K most similar chunks for a natural language question.

        Args:
            question: The user's query text.

        Returns:
            A list of retrieved chunks containing the metadata and distance score.
        """
        if not question or not question.strip():
            self.logger.error("Empty question provided to RetrievalService.")
            raise ValueError("Question must not be empty.")

        question_text = question.strip()
        self.logger.info("Retrieving chunks for question: %s", question_text)

        try:
            response = self.embedding_service.client.embeddings.create(
                model=self.embedding_service.model_name,
                input=question_text,
            )
            question_embedding = response.data[0].embedding
        except OpenAIError as exc:
            self.logger.exception("Failed to embed question for retrieval.")
            raise RuntimeError("Failed to create embedding for the question.") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error during question embedding.")
            raise RuntimeError("Failed to embed the question.") from exc

        try:
            query_results = self.vector_store_service.query_chunks(
                query_embedding=question_embedding,
                top_k=self.top_k,
            )
        except Exception as exc:
            self.logger.exception("ChromaDB query failed for question retrieval.")
            raise RuntimeError("Failed to query ChromaDB for retrieval.") from exc

        if not query_results or not query_results.get('documents'):
            self.logger.info("No chunks were retrieved for the question.")
            return []

        retrieved_chunks: List[RetrievedChunk] = []
        documents = query_results.get('documents', [[]])[0]
        metadatas = query_results.get('metadatas', [[]])[0]
        distances = query_results.get('distances', [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            chunk_id = metadata.get('chunk_id', '')
            page_number = int(metadata.get('page_number', 0))
            chunk_number = int(metadata.get('chunk_number', 0))
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    chunk_number=chunk_number,
                    page_number=page_number,
                    text=text,
                    distance=float(distance),
                )
            )

        return retrieved_chunks
