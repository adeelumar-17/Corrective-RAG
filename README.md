# 📚 Corrective RAG Assistant

A self-correcting Retrieval-Augmented Generation system that grades retrieved chunks for relevance and falls back to web search when documents can't answer the question.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple)

---

## Architecture

```
User Question
     │
     ▼
┌──────────┐
│ RETRIEVE │  Search Pinecone (MMR) → top 5 diverse chunks
└────┬─────┘
     ▼
┌──────────┐
│  GRADE   │  LLM evaluates each chunk: relevant? yes/no
└────┬─────┘
     │
  ┌──┴──┐
  │     │
  ▼     ▼
 YES    NO
  │     │
  │  ┌──┴───────┐
  │  │WEB SEARCH│  Tavily API → 3 web results
  │  └──┬───────┘
  │     │
  ▼     ▼
┌──────────┐
│ GENERATE │  Groq LLM → final answer with source attribution
└──────────┘
     │
     ▼
📄 From Docs  or  🌐 From Web
```

## Features

- **PDF Upload** — Drag-and-drop multiple PDFs; text is extracted, chunked, and stored
- **Smart Retrieval** — MMR search returns diverse, relevant chunks (not duplicates)
- **LLM Grading** — Each chunk is evaluated for relevance before generating
- **Web Fallback** — Automatically searches the web when documents can't help
- **Source Attribution** — Every answer shows whether it came from docs 📄 or web 🌐
- **Expandable Sources** — Click to see exactly which pages/URLs were used
- **Premium UI** — Dark glassmorphism theme with smooth animations
- **REST API** — Clean FastAPI backend; any frontend can consume it

## Tech Stack

| Component | Tool |
|-----------|------|
| **LLM** | Groq API (LLaMA 3.3 70B) |
| **Embeddings** | HuggingFace BGE-small-en-v1.5 (local) |
| **Vector DB** | Pinecone (cloud, serverless) |
| **Search** | MMR (Maximal Marginal Relevance) |
| **Web Search** | Tavily API |
| **Agent Framework** | LangGraph |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Deployment** | Render |

## Project Structure

```
corrective-rag/
├── main.py                 # FastAPI backend (API endpoints + static serving)
├── rag/
│   ├── __init__.py         # Config & shared constants
│   ├── ingestor.py         # PDF → chunks → embeddings → Pinecone
│   ├── retriever.py        # Pinecone MMR search
│   ├── grader.py           # LLM relevance grading
│   ├── web_search.py       # Tavily web search fallback
│   ├── generator.py        # Groq LLM answer generation
│   └── graph.py            # LangGraph CRAG state machine
├── static/
│   ├── index.html          # Frontend page
│   ├── style.css           # Dark theme styling
│   └── script.js           # Chat logic & API calls
├── requirements.txt
├── .env.example
├── LEARNING_GUIDE.md       # Detailed concept explanations
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/corrective-rag.git
cd corrective-rag
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get free API keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| Groq | [console.groq.com](https://console.groq.com) | Generous free tier |
| Tavily | [tavily.com](https://tavily.com) | 1,000 searches/month |
| Pinecone | [pinecone.io](https://www.pinecone.io/) | 1 index, ~100K vectors |

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 6. Run the app

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> **Note:** The embedding model (~130MB) downloads on first run. You'll see this happen in the terminal.

## Deploy to Render

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/corrective-rag.git
   git push -u origin main
   ```

2. **Create Render Web Service**
   - Go to [render.com](https://render.com) → New Web Service
   - Connect your GitHub repo
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables** (in Render dashboard → Environment)
   ```
   GROQ_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   ```

4. **Deploy** — Your app will be live at `https://your-app.onrender.com`

> **✅ Pinecone advantage:** Since Pinecone is cloud-hosted, your uploaded documents persist across deploys — no re-upload needed!

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload PDF files (multipart/form-data) |
| `POST` | `/api/query` | Ask a question (`{"question": "..."}`) |
| `GET` | `/api/status` | Check document count |
| `DELETE` | `/api/clear` | Remove all vectors from Pinecone |
| `GET` | `/` | Serve the frontend |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key for LLM (grading + generation) |
| `TAVILY_API_KEY` | ✅ | Tavily API key for web search fallback |
| `PINECONE_API_KEY` | ✅ | Pinecone API key for vector storage |

## License

MIT
