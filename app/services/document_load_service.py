"""Service for loading text from PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class DocumentLoadService:
    """Load and extract text from a PDF file page by page."""

    def __init__(self) -> None:
        self.logger = logger

    def load_pdf(self, file_path: str | Path) -> List[Tuple[int, str]]:
        """Extract text from each page of a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of tuples containing the page number and extracted text.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a PDF or cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            self.logger.error("PDF file not found: %s", path)
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            self.logger.error("Unsupported file type: %s", path)
            raise ValueError(f"Unsupported file type: {path}")

        try:
            reader = PdfReader(str(path))
            pages: List[Tuple[int, str]] = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append((page_number, text.strip()))

            self.logger.info("Successfully loaded %d page(s) from %s", len(pages), path)
            return pages
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to parse PDF file: %s", path)
            raise ValueError(f"Failed to parse PDF file: {path}") from exc
