"""Data model for a document chunk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentChunk:
    """Represents a chunk of text extracted from a PDF page."""

    chunk_number: int
    page_number: int
    text: str
