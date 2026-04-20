"""
Corrective RAG — FastAPI Backend (API Only)

This is the main backend entry point. It provides:
  - REST API endpoints for document upload, querying, status, and clearing
  - CORS middleware for cross-origin requests (React frontend on Vercel)

IMPORTANT: Heavy ML modules (sentence-transformers, torch, langchain) are
imported lazily inside endpoint functions — NOT at startup. This ensures
uvicorn binds the port fast enough for Render's health check.

Run locally:
    uvicorn main:app --reload

Deploy on Render:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env (must happen before any rag imports)
load_dotenv()

# Only import lightweight config — no heavy ML deps here
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = "rag-docs"


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Corrective RAG Assistant",
    description="A self-correcting RAG system with LangGraph, Pinecone, and Gemini",
    version="1.0.0",
)

# Allow frontend to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models for request/response validation
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    used_web_search: bool


# ---------------------------------------------------------------------------
# API Endpoints (heavy imports are LAZY — inside the functions)
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF files.

    Accepts multiple PDF files via multipart/form-data.
    Extracts text, chunks, embeds, and stores in Pinecone.
    Returns the number of chunks created.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate all files are PDFs
    pdf_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF file",
            )
        content = await file.read()
        pdf_files.append((file.filename, content))

    # Lazy import — only loads heavy ML modules on first upload
    try:
        from rag.ingestor import ingest_pdfs

        chunk_count = ingest_pdfs(pdf_files)
        return {
            "message": "Documents processed successfully",
            "chunk_count": chunk_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question and get an answer from the CRAG pipeline.

    The system will:
    1. Search Pinecone for relevant document chunks
    2. Grade chunks for relevance (LLM-based)
    3. If relevant → generate answer from documents
    4. If irrelevant → fall back to web search → generate answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        from rag.graph import run_graph

        result = run_graph(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/status")
async def status():
    """
    Check whether documents are loaded in Pinecone.

    Returns the total number of vectors (chunks) in the index.
    """
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME in existing_indexes:
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            return {
                "docs_loaded": total_vectors > 0,
                "chunk_count": total_vectors,
            }

        return {"docs_loaded": False, "chunk_count": 0}

    except Exception:
        return {"docs_loaded": False, "chunk_count": 0}


@app.delete("/api/clear")
async def clear_database():
    """
    Delete all vectors from the Pinecone index.

    This removes all uploaded document data but keeps the index itself.
    """
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME in existing_indexes:
            index = pc.Index(PINECONE_INDEX_NAME)
            index.delete(delete_all=True)

        return {"message": "Database cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@app.get("/")
async def root():
    """Health check / API info."""
    return {
        "app": "Corrective RAG Assistant",
        "status": "running",
        "docs": "/docs",
    }
