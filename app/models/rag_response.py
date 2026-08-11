"""Models for RAG service responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class Source:
    """Represents a source chunk used to produce the RAG answer."""

    chunk_id: str
    chunk_number: int
    page_number: int
    distance: float


@dataclass(slots=True)
class RagResponse:
    """Structured response from the RAGService.

    Attributes:
        answer: The generated answer text from the LLM.
        sources: List of `Source` objects describing which chunks were used.
    """

    answer: str
    sources: List[Source]
