"""Simple embedding service for document chunks using the OpenAI embeddings API."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

from app.models.document_chunk import DocumentChunk
from app.models.embedded_document_chunk import EmbeddedDocumentChunk


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class EmbeddingService:
    """Generate embeddings for document chunks using OpenAI."""

    def __init__(self, client: Optional[OpenAI] = None, model_name: Optional[str] = None) -> None:
        self.client = client or self._create_client()
        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.logger = logger

    def _create_client(self) -> OpenAI:
        """Create an OpenAI client using the configured API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in the environment.")

        return OpenAI(api_key=api_key)

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> List[EmbeddedDocumentChunk]:
        """Generate an embedding for each document chunk.

        Args:
            chunks: A sequence of document chunks to embed.

        Returns:
            A list of embedded document chunks.
        """
        if not chunks:
            self.logger.info("No chunks provided for embedding.")
            return []

        embedded_chunks: List[EmbeddedDocumentChunk] = []

        for chunk in chunks:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk.text,
                )
                embedding = response.data[0].embedding
                embedded_chunks.append(
                    EmbeddedDocumentChunk(
                        chunk_number=chunk.chunk_number,
                        page_number=chunk.page_number,
                        text=chunk.text,
                        embedding=embedding,
                    )
                )
            except OpenAIError as exc:
                self.logger.exception("OpenAI embedding request failed for chunk %s", chunk.chunk_number)
                raise RuntimeError(f"Failed to embed chunk {chunk.chunk_number}") from exc
            except Exception as exc:
                self.logger.exception("Unexpected error while embedding chunk %s", chunk.chunk_number)
                raise RuntimeError(f"Failed to embed chunk {chunk.chunk_number}") from exc

        return embedded_chunks
