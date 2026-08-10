"""Simple fixed-size chunking service for PDF page text."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from app.models.document_chunk import DocumentChunk


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class ChunkingService:
    """Split page text into fixed-size chunks with overlap."""

    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 25) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger

    def chunk_pages(self, pages: Sequence[Tuple[int, str]]) -> List[DocumentChunk]:
        """Create chunks for each page using fixed-size character windows.

        Args:
            pages: Sequence of ``(page_number, text)`` tuples from the document loader.

        Returns:
            A list of ``DocumentChunk`` objects.
        """
        chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page_number, text in pages:
            if not text or not text.strip():
                self.logger.info("Skipping empty content for page %s", page_number)
                continue

            chunk_counter = self._chunk_page_text(page_number, text, chunks, chunk_counter)

        self.logger.info("Created %d chunk(s) from %d page(s)", len(chunks), len(pages))
        return chunks

    def _chunk_page_text(
        self,
        page_number: int,
        text: str,
        chunks: List[DocumentChunk],
        start_counter: int,
    ) -> int:
        """Split a single page's text into fixed-size chunks with overlap."""
        index = 0
        chunk_counter = start_counter

        while index < len(text):
            end_index = min(index + self.chunk_size, len(text))
            chunk_text = text[index:end_index].strip()

            if chunk_text:
                chunk_counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_number=chunk_counter,
                        page_number=page_number,
                        text=chunk_text,
                    )
                )

            if end_index >= len(text):
                break

            index += self.chunk_size - self.chunk_overlap

        return chunk_counter
