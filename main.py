"""Simple CLI entry point for loading a sample PDF, chunking it, and generating embeddings."""

from __future__ import annotations

from pathlib import Path

from app.services.chunking_service import ChunkingService
from app.services.document_load_service import DocumentLoadService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


def main() -> None:
    """Run the basic Naive RAG ingestion pipeline and persist embedded chunks in ChromaDB."""
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "documents" / "naive_rag_sample_5_page.pdf"

    document_service = DocumentLoadService()
    pages = document_service.load_pdf(pdf_path)

    print(f"Loaded {len(pages)} page(s) from {pdf_path}")
    print("Sample page text (first 160 characters):")

    for page_number, text in pages:
        print(f"\n--- Page {page_number} ---")
        print(f"Text: {text[:160]}{'...' if len(text) > 160 else ''}")

    chunking_service = ChunkingService(chunk_size=300, chunk_overlap=30)
    chunks = chunking_service.chunk_pages(pages)

    print(f"Created {len(chunks)} chunk(s) from {len(pages)} page(s)")
    print("Sample chunk text (first 160 characters):")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_number} ---")
        print(f"Page: {chunk.page_number}")
        print(f"Text: {chunk.text[:160]}{'...' if len(chunk.text) > 160 else ''}")

    embedding_service = EmbeddingService()
    embedded_chunks = embedding_service.embed_chunks(chunks)
    print(f"Generated embeddings for {len(embedded_chunks)} chunk(s)")
    for chunk in embedded_chunks:
        preview = chunk.embedding[:6]
        print(f"\n--- Chunk {chunk.chunk_number} ---")
        print(f"Page: {chunk.page_number}")
        print(f"Text: {chunk.text[:160]}{'...' if len(chunk.text) > 160 else ''}")
        print(f"Embedding dimensions: {len(chunk.embedding)}")
        print(f"Embedding preview: {preview}")

    vector_store_service = VectorStoreService()
    vector_store_service.add_chunks(embedded_chunks)


    print(f"Loaded {len(pages)} page(s) from {pdf_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total embeddings generated: {len(embedded_chunks)}")
    print(f"Total vectors stored in ChromaDB: {vector_store_service.count()}")

if __name__ == "__main__":
    main()
