import shutil
from pathlib import Path

import pytest

from app.models.embedded_document_chunk import EmbeddedDocumentChunk
from app.services.vector_store_service import VectorStoreService


@pytest.fixture
def vector_store_service(tmp_path: Path) -> VectorStoreService:
    storage_path = tmp_path / "chroma"
    return VectorStoreService(persist_directory=str(storage_path), collection_name="test_collection")


def test_add_chunks_stores_documents_and_metadata(vector_store_service: VectorStoreService) -> None:
    embedded_chunks = [
        EmbeddedDocumentChunk(
            chunk_number=1,
            page_number=1,
            text="First chunk",
            embedding=[0.1, 0.2, 0.3],
        ),
        EmbeddedDocumentChunk(
            chunk_number=2,
            page_number=2,
            text="Second chunk",
            embedding=[0.4, 0.5, 0.6],
        ),
    ]

    vector_store_service.add_chunks(embedded_chunks)

    assert vector_store_service.count() == 2

    collection = vector_store_service.collection
    stored = collection.get(include=["metadatas", "documents"])
    assert len(stored["ids"]) == 2
    assert stored["documents"] == ["First chunk", "Second chunk"]
    assert stored["metadatas"][0]["chunk_number"] == 1
    assert stored["metadatas"][1]["page_number"] == 2
