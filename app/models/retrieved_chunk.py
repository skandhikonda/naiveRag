"""Data model for a chunk retrieved from the vector store."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    """Represents a chunk retrieved from ChromaDB."""

    chunk_id: str
    chunk_number: int
    page_number: int
    text: str
    distance: float
