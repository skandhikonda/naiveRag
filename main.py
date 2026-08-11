"""Simple CLI entry point for loading a sample PDF, chunking it, and generating embeddings."""

from __future__ import annotations

from pathlib import Path

from app.services.chunking_service import ChunkingService
from app.services.document_load_service import DocumentLoadService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService


def main() -> None:
    """Run the Naive RAG ingestion pipeline and retrieve chunks for a user question."""
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

    question = input("\nEnter a question for retrieval: ").strip()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        top_k=3,
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        embedding_service=embedding_service,
    )

    print("\n==================================================")
    print("Question:")
    print(question)
    print("\n==================================================")

    try:
        rag_response = rag_service.answer(question)
    except Exception as exc:
        print(f"Failed to generate answer: {exc}")
        return

    print("\nAnswer:\n")
    print(rag_response.answer)

    print("\nSources:\n")
    if not rag_response.sources:
        print("- No sources available")
    else:
        for src in rag_response.sources:
            print(f"- Page {src.page_number}, Chunk {src.chunk_number}, Distance: {src.distance}")

if __name__ == "__main__":
    main()
