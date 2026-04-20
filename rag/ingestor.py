"""
Ingestor Module — PDF → Chunks → Embeddings → Pinecone

This module handles the document ingestion pipeline:
1. Reads PDF files and extracts text page by page
2. Splits text into overlapping chunks using RecursiveCharacterTextSplitter
3. Generates embeddings using the BGE model (runs locally)
4. Stores the embedded chunks in Pinecone (cloud vector database)
"""

import io
from typing import List, Tuple

from pypdf import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ---------------------------------------------------------------------------
# Singleton: Embedding model (downloads ~130MB on first run, then cached)
# ---------------------------------------------------------------------------
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a shared embedding model instance.

    The model is loaded once and reused across all calls to avoid
    re-downloading and re-initializing on every request.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            # normalize_embeddings=True is recommended by BGE authors for cosine similarity
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Pinecone index setup
# ---------------------------------------------------------------------------
def ensure_pinecone_index() -> None:
    """
    Create the Pinecone index if it doesn't already exist.

    Uses serverless spec (free tier) with cosine similarity,
    which pairs well with the normalized BGE embeddings.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # 384 for BGE-small
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------
def ingest_pdfs(pdf_files: List[Tuple[str, bytes]]) -> int:
    """
    Process uploaded PDF files and store their chunks in Pinecone.

    Args:
        pdf_files: List of (filename, file_bytes) tuples.
                   Each tuple contains the original filename and the raw
                   bytes of the PDF file (read from FastAPI's UploadFile).

    Returns:
        The number of chunks stored in Pinecone.

    Pipeline:
        PDF bytes → pypdf text extraction → RecursiveCharacterTextSplitter →
        HuggingFaceEmbeddings → PineconeVectorStore
    """

    # ---- Step 1: Extract text from each PDF page ----
    all_documents: List[Document] = []

    for filename, file_bytes in pdf_files:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            print(f"[Ingestor] Failed to read {filename}: {e}")
            continue

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,  # 1-indexed for human readability
                    },
                )
                all_documents.append(doc)

    if not all_documents:
        return 0

    # ---- Step 2: Split into chunks ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Split priority order
    )
    chunks = text_splitter.split_documents(all_documents)

    if not chunks:
        return 0

    # ---- Step 3: Ensure Pinecone index exists ----
    ensure_pinecone_index()

    # ---- Step 4: Embed and upsert to Pinecone ----
    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    return len(chunks)
