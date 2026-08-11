"""Simple vector store service for persisting embedded document chunks in ChromaDB."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional, Sequence

import chromadb

from app.models.embedded_document_chunk import EmbeddedDocumentChunk


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class VectorStoreService:
    """Store embedded document chunks in a local persistent ChromaDB collection."""

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "naive_rag_documents") -> None:
        """Initialize the persistent ChromaDB client and collection.

        Args:
            persist_directory: Optional directory where ChromaDB should persist its data.
            collection_name: Name of the Chroma collection to use or create.
        """
        default_directory = Path(__file__).resolve().parents[2] / "data" / "chroma"
        self.persist_directory = Path(persist_directory) if persist_directory else default_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.logger = logger

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.exception("Failed to initialize ChromaDB at %s", self.persist_directory)
            raise RuntimeError("Unable to initialize ChromaDB storage.") from exc

    def add_chunks(self, chunks: Sequence[EmbeddedDocumentChunk]) -> None:
        """Store embedded chunks in ChromaDB.

        Args:
            chunks: Embedded document chunks to persist.
        """
        if not chunks:
            self.logger.info("No chunks provided; skipping ChromaDB storage.")
            return

        try:
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            documents = [chunk.text for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            metadatas = [
                {
                    "chunk_id": chunk_id,
                    "chunk_number": chunk.chunk_number,
                    "page_number": chunk.page_number,
                }
                for chunk_id, chunk in zip(chunk_ids, chunks)
            ]

            self.collection.add(
                ids=chunk_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self.logger.info(
                "Stored %s chunk(s) in ChromaDB collection '%s'",
                len(chunks),
                self.collection_name,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.exception("Failed to store chunks in ChromaDB collection '%s'", self.collection_name)
            raise RuntimeError(f"Failed to store chunks in ChromaDB collection '{self.collection_name}'.") from exc

    def count(self) -> int:
        """Return the number of stored chunks in the collection."""
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.exception("Failed to count stored chunks in ChromaDB collection '%s'", self.collection_name)
            raise RuntimeError(f"Unable to count stored chunks in ChromaDB collection '{self.collection_name}'.") from exc

    def query_chunks(self, query_embedding: list[float], top_k: int = 3) -> dict:
        """Query the ChromaDB collection for the most similar chunks.

        Args:
            query_embedding: Embedding vector for the query.
            top_k: Number of similar chunks to retrieve.

        Returns:
            The raw query result from ChromaDB.
        """
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.logger.exception("Failed to query ChromaDB collection '%s'", self.collection_name)
            raise RuntimeError(f"Unable to query ChromaDB collection '{self.collection_name}'.") from exc
