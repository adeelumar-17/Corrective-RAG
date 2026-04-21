# Corrective RAG

Self-correcting Retrieval-Augmented Generation (CRAG) with a FastAPI backend and React frontend.

The system retrieves context from uploaded PDFs, grades relevance with an LLM, and automatically falls back to web search when local documents are not enough.

## What Makes It "Corrective"

Classic RAG can generate answers from weak context. This project adds a correction step:

1. Retrieve top document chunks from Pinecone (MMR)
2. Grade each chunk for relevance using Groq LLM
3. If too many chunks are irrelevant, trigger Tavily web search
4. Generate the final answer with explicit sources

This reduces hallucinations when uploaded documents do not cover the question.

## Architecture

```text
Question
  -> Retrieve from Pinecone (MMR)
  -> Grade chunk relevance (yes/no)
     -> relevant enough: Generate from docs
     -> not relevant enough: Web search (Tavily) -> Generate
  -> Return answer + sources + web_fallback flag
```

## Tech Stack

- Backend: FastAPI + Uvicorn
- Orchestration: LangGraph
- LLM: Groq (`llama-3.3-70b-versatile`)
- Embeddings: HuggingFace BGE small (`BAAI/bge-small-en-v1.5`, 384-dim, local CPU)
- Vector DB: Pinecone Serverless
- Web fallback: Tavily Search
- Frontend: React + Vite

## Project Structure

```text
.
├── main.py                    # FastAPI API entry point
├── requirements.txt
├── Dockerfile
├── LEARNING_GUIDE.md
├── rag/
│   ├── __init__.py            # Env + constants
│   ├── ingestor.py            # PDF -> chunks -> embeddings -> Pinecone
│   ├── retriever.py           # Pinecone MMR retrieval
│   ├── grader.py              # LLM relevance grading
│   ├── web_search.py          # Tavily fallback search
│   ├── generator.py           # Final answer generation + sources
│   └── graph.py               # LangGraph CRAG workflow
└── frontend/
    ├── package.json
    ├── vite.config.js         # Dev proxy /api -> localhost:8000
    ├── vercel.json
    └── src/
        ├── App.jsx
        ├── api.js
        └── components/
```

## API Endpoints

- `POST /api/upload`: Upload one or more PDF files (`multipart/form-data`, field name `files`)
- `POST /api/query`: Ask a question (`{"question": "..."}`)
- `GET /api/status`: Check if vectors are loaded and get chunk count
- `DELETE /api/clear`: Clear all vectors from Pinecone index
- `GET /`: API health/info response

## Quick Start (Local)

### 1) Clone and set up Python environment

```bash
git clone https://github.com/<your-username>/Corrective-RAG.git
cd Corrective-RAG

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
```

### 3) Run backend

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.

### 4) Run frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

The Vite dev server proxies `/api/*` requests to `http://localhost:8000`.

## How Retrieval Works

- PDFs are parsed page-by-page with `pypdf`
- Text is chunked using:
  - chunk size: `500`
  - chunk overlap: `100`
- Chunks are embedded locally with BGE small
- Stored in Pinecone index `rag-docs` (cosine similarity)
- Retriever uses MMR (`k=5`, `fetch_k=20`) for diverse context

## Response Format

`POST /api/query` returns:

```json
{
  "answer": "...",
  "sources": ["file.pdf (page 3)", "https://..."],
  "used_web_search": false
}
```

## Deployment Notes

### Backend

- The repository includes a `Dockerfile` and is compatible with container-based deployment (for example, Hugging Face Spaces Docker mode or similar platforms).
- Set `GROQ_API_KEY`, `TAVILY_API_KEY`, and `PINECONE_API_KEY` in your deployment environment.

### Frontend

- `frontend/` is ready for Vercel deployment.
- Set `VITE_API_URL` to your deployed backend URL so production requests target the API.

## Known Behavior

- First ingestion may take longer because the embedding model is downloaded and cached locally.
- Frontend session logic clears Pinecone once per new browser session to prevent free-tier accumulation during iterative demos.

## Learning Resource

For a beginner-friendly deep dive into the concepts and file-by-file explanation, read `LEARNING_GUIDE.md`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

## License

No LICENSE file is currently included. Add your preferred license (for example, MIT) before open-source distribution.
