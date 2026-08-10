"""Data model for a document chunk with an embedding vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class EmbeddedDocumentChunk:
    """Represents a chunk of text together with its embedding vector."""

    chunk_number: int
    page_number: int
    text: str
    embedding: List[float]
