"""
Corrective RAG — Configuration & Shared Constants

All API keys and model settings are centralized here.
Each module imports from this package instead of managing its own config.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
# On Render, these are set via the dashboard instead
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Set API keys as env vars so LangChain integrations can auto-detect them
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Set TAVILY_API_KEY as env var so langchain-community tools can auto-detect it
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# --- Model Settings ---
LLM_MODEL = "gemini-2.0-flash"                # Google Gemini model for grading & generation
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # Local HuggingFace embedding model (384 dims)
EMBEDDING_DIMENSION = 384                     # Output dimension of BGE-small

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "rag-docs"              # Name of the Pinecone index
PINECONE_CLOUD = "aws"                        # Cloud provider for serverless
PINECONE_REGION = "us-east-1"                 # Region for serverless

# --- Chunking Settings ---
CHUNK_SIZE = 500                              # Characters per chunk
CHUNK_OVERLAP = 100                           # Overlap between adjacent chunks

# --- Retrieval Settings ---
RETRIEVAL_K = 5                               # Number of chunks to return
RETRIEVAL_FETCH_K = 20                        # Candidate pool size for MMR
