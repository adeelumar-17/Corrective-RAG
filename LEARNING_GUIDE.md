# 📚 Corrective RAG — Complete Learning Guide & Project Plan

> **Who is this for?** You — someone building their first RAG application. This guide explains **every concept from scratch** before we write a single line of code.

---

## Table of Contents

1. [The Big Picture — What Are We Building?](#1-the-big-picture)
2. [Core Concepts You Need to Know](#2-core-concepts)
   - 2.1 What is RAG?
   - 2.2 What is Corrective RAG (CRAG)?
   - 2.3 Embeddings — Turning Text Into Numbers
   - 2.4 Vector Databases (ChromaDB)
   - 2.5 Chunking — Why We Split Documents
   - 2.6 LLMs & Groq
   - 2.7 LangChain — The Glue
   - 2.8 LangGraph — The Brain (State Machines)
   - 2.9 Tavily — Web Search Fallback
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
5. [Implementation Order & Learning Path](#5-implementation-order--learning-path)
6. [Glossary](#6-glossary)
7. [Common Pitfalls & Debugging Tips](#7-common-pitfalls--debugging-tips)
8. [Resources & Further Reading](#8-resources--further-reading)

---

## 1. The Big Picture

### What Are We Building?

Imagine you have a stack of PDF documents — lecture notes, research papers, manuals. You want to ask questions and get accurate answers **from those documents**. That's RAG.

But here's the problem with basic RAG: sometimes the retrieved chunks of text **aren't actually relevant** to your question. A basic RAG system would still try to generate an answer from irrelevant chunks, leading to hallucinated or wrong answers.

**Corrective RAG (CRAG)** fixes this by adding a **grading step**. After retrieving chunks, an LLM checks: *"Are these chunks actually relevant to the question?"*

- ✅ **If relevant** → Generate the answer from the documents
- ❌ **If irrelevant** → Fall back to a **web search** and answer from the web instead

This makes the system **self-correcting** — it knows when it doesn't have good information and goes looking for better sources.

### The User Experience

```
┌─────────────────────────────────────────────────┐
│  User uploads PDFs  ──►  System processes them  │
│                                                 │
│  User asks: "What is backpropagation?"          │
│                                                 │
│  System retrieves relevant chunks from PDFs     │
│  System grades: "Are these chunks relevant?" ✅  │
│  System generates answer from documents         │
│  Shows: 📄 Answered from your documents         │
│                                                 │
│  User asks: "What's the weather in Lahore?"     │
│                                                 │
│  System retrieves chunks (none are relevant)    │
│  System grades: "Are these chunks relevant?" ❌  │
│  System falls back to web search                │
│  Shows: 🌐 Answered from web search             │
└─────────────────────────────────────────────────┘
```

---

## 2. Core Concepts

### 2.1 What is RAG? (Retrieval-Augmented Generation)

**The Problem:**
LLMs (like ChatGPT, LLaMA) are trained on general internet data up to a certain date. They:
- Don't know about **your private documents** (your PDFs, notes, etc.)
- Can **hallucinate** (make up facts confidently)
- Have a **knowledge cutoff** (don't know recent events)

**The Solution — RAG:**
Instead of relying only on the LLM's training data, we **retrieve** relevant information from our own documents and feed it to the LLM along with the question. The LLM then generates an answer based on the **retrieved context**.

```
Traditional LLM:
  Question ──► LLM ──► Answer (may hallucinate)

RAG:
  Question ──► Search your docs ──► Get relevant chunks ──► LLM + chunks ──► Answer (grounded in your data)
```

**Why "Retrieval-Augmented"?**
- **Retrieval**: We search/retrieve relevant pieces of information
- **Augmented**: We augment (enhance) the LLM's input with this information
- **Generation**: The LLM generates the final answer

> [!NOTE]
> **Analogy:** Think of it like an open-book exam. Instead of relying on memory alone (the LLM), you look up the relevant pages in your textbook (retrieval) and then write your answer (generation).

### 2.2 What is Corrective RAG (CRAG)?

Standard RAG has a weakness: it **always** tries to answer from retrieved documents, even if those documents aren't relevant to the question. CRAG adds an intelligence layer:

```
Standard RAG Pipeline:
  Question → Retrieve → Generate (always from docs, even if irrelevant)

Corrective RAG Pipeline:
  Question → Retrieve → GRADE (are chunks relevant?) 
                              ├── YES → Generate from docs  📄
                              └── NO  → Web Search → Generate from web  🌐
```

**The "grading" step** is the key innovation. An LLM looks at each retrieved chunk and the question, then decides: *"Is this chunk useful for answering this question?"*

**The grading logic:**
1. Retrieve 5 chunks from your documents
2. Send each chunk + the question to the LLM grader
3. LLM responds "yes" (relevant) or "no" (irrelevant) for each chunk
4. **Decision rule:**
   - If **more than half** are irrelevant (or zero docs exist) → trigger web search
   - Otherwise → use the relevant chunks to generate the answer

> [!IMPORTANT]
> **Why is this important?** Without grading, a question like "What's the weather today?" would get random PDF chunks and the LLM would hallucinate an answer. With CRAG, the grader says "none of these chunks are about weather" and the system correctly falls back to web search.

### 2.3 Embeddings — Turning Text Into Numbers

This is one of the most important concepts. Computers can't "understand" text directly — they work with numbers. **Embeddings** convert text into numerical vectors (lists of numbers) that capture the **meaning** of the text.

**How it works conceptually:**

```
"The cat sat on the mat"  →  [0.23, -0.45, 0.78, 0.12, ..., -0.33]   (384 numbers)
"A kitten rested on a rug" →  [0.21, -0.43, 0.76, 0.14, ..., -0.31]   (384 numbers)
"Python is a programming language" → [0.89, 0.12, -0.67, 0.45, ..., 0.22]   (384 numbers)
```

Notice:
- The first two sentences have **similar meanings** → their vectors are **close together** (similar numbers)
- The third sentence is about something completely different → its vector is **far away**

**The embedding model we'll use:** `BAAI/bge-small-en-v1.5`
- "BGE" = BAAI General Embedding
- "small" = lightweight version (~130MB), runs on CPU
- "en" = English language
- "v1.5" = version 1.5
- Produces vectors of **384 dimensions** (384 numbers per text chunk)
- Runs **locally** — no API calls needed, completely free

> [!TIP]
> **Analogy:** Imagine embeddings as GPS coordinates for text. Just like similar locations have close GPS coordinates, similar texts have close embedding vectors. When you search, you're essentially asking "which text coordinates are closest to my question's coordinates?"

**Why HuggingFace?**
HuggingFace is like the "GitHub for ML models." It hosts pre-trained models that you can download and use. The `sentence-transformers` library makes it easy to use these models for generating embeddings.

### 2.4 Vector Databases (Pinecone)

Once you have embeddings (numerical vectors), you need somewhere to **store** them and **search** through them efficiently. That's what a vector database does.

**Pinecone** is a **cloud-hosted**, fully managed vector database. Unlike local databases, your data lives on Pinecone's servers — accessible from anywhere, no local storage needed.

```
┌─────────────────────────────────────────────────────────────┐
│  Pinecone Index: "rag-docs"                                 │
│  (hosted in the cloud)                                      │
│                                                             │
│  ID   │ Text Chunk (metadata)     │ Embedding    │ Metadata │
│  ─────┼───────────────────────────┼──────────────┼──────────│
│  1    │ "Backprop computes..."    │ [0.2, -0.4]  │ {src: "ml.pdf", page: 5}  │
│  2    │ "Neural networks are..."  │ [0.3, -0.3]  │ {src: "ml.pdf", page: 6}  │
│  3    │ "SQL is a query lang..."  │ [0.8, 0.1]   │ {src: "db.pdf", page: 1}  │
└─────────────────────────────────────────────────────────────┘
```

**How search works (similarity search):**

1. Your question "What is backpropagation?" gets embedded → `[0.19, -0.41, ...]`
2. Pinecone compares this vector against ALL stored vectors (in the cloud)
3. Returns the **top K** most similar vectors (we use K=5)
4. Those correspond to the most relevant text chunks

**Key Pinecone concepts:**
- **Index**: Like a table in a regular database. We create one index called `"rag-docs"`
- **Namespace**: A partition inside an index — we can use it to separate different users' documents
- **Cloud-hosted**: Data persists permanently in the cloud — no ephemeral storage worries on Render!
- **Metadata**: Extra info attached to each vector (filename, page, text content) — used for source attribution
- **Free tier**: 1 index, ~100K vectors, enough for our project. Sign up at [pinecone.io](https://www.pinecone.io/)

**Why Pinecone over ChromaDB?**
- **Cloud-native**: No local files to manage — perfect for deployment (Render's filesystem is ephemeral)
- **Scales automatically**: Handles millions of vectors without config changes
- **Always available**: Data persists across app restarts and redeployments
- **Better search**: Supports advanced similarity metrics and metadata filtering

> [!NOTE]
> **Why not a regular database (like PostgreSQL)?** Regular databases search by exact matches (e.g., find rows where name = "John"). Vector databases search by **meaning similarity** — they find text that is **semantically close** to your query, even if the exact words differ.

### 2.4.1 Maximal Marginal Relevance (MMR) — Better Similarity Search

Basic similarity search has a problem: the top 5 results might all say essentially the **same thing** (redundant chunks from nearby paragraphs). **MMR** fixes this.

**How MMR works:**
1. Find the top ~20 most similar chunks (the candidate pool)
2. Pick the single most relevant one
3. For each remaining pick, balance between:
   - **Relevance** to the query (high similarity = good)
   - **Diversity** from already-picked chunks (low similarity to picks = good)
4. Repeat until you have K results

```
Plain Top-K:  [chunk about backprop] [chunk about backprop] [chunk about backprop] [chunk about backprop] [chunk about backprop]
                 ↑ All very similar — redundant!

MMR Top-K:    [chunk about backprop] [chunk about gradient descent] [chunk about chain rule] [chunk about loss functions] [chunk about weight updates]
                 ↑ All relevant, but each adds NEW information!
```

> [!TIP]
> **Analogy:** Imagine searching for "best restaurants in Lahore". Plain search might give you 5 reviews of the same restaurant. MMR gives you one review each of the 5 best different restaurants — much more useful!

We'll use MMR via LangChain's retriever with `search_type="mmr"` — it works seamlessly with Pinecone.

### 2.5 Chunking — Why We Split Documents

You can't embed an entire 50-page PDF as one vector — the embedding would be too vague and lose important details. So we split documents into smaller **chunks**.

**RecursiveCharacterTextSplitter** (from LangChain):

This splitter tries to keep text coherent by splitting on natural boundaries in this priority order:
1. Double newlines (`\n\n`) — paragraph breaks
2. Single newlines (`\n`) — line breaks
3. Spaces (` `) — word boundaries
4. Characters — last resort

**Our settings:**
```
chunk_size = 500     # Each chunk is ~500 characters (roughly 100 words)
chunk_overlap = 100  # Adjacent chunks share 100 characters of overlap
```

**Why overlap?** Imagine a sentence that spans two chunks. Without overlap, important context could be split between chunks and lost. Overlap ensures that boundary information is captured in both adjacent chunks.

```
Original text: "...gradient descent. Backpropagation computes the gradient of the loss function. This gradient is then used to update weights..."

Chunk 1: "...gradient descent. Backpropagation computes the gradient of the loss function."
Chunk 2: "Backpropagation computes the gradient of the loss function. This gradient is then used to update weights..."
                    ───────────────── OVERLAP ─────────────────
```

> [!TIP]
> **Chunk size trade-off:**
> - Too small (100 chars) → Chunks lack context, retrieval is noisy
> - Too large (2000 chars) → Embeddings become vague, search is less precise
> - 500 chars is a solid starting point for most use cases

### 2.6 LLMs & Groq

**LLM (Large Language Model):**
A neural network trained on massive amounts of text that can generate human-like responses. We use `llama-3.3-70b-versatile` — Meta's LLaMA 3.3 model with 70 billion parameters.

**Groq:**
Groq is a company that provides **extremely fast** LLM inference (generating responses). They have custom hardware (LPUs — Language Processing Units) that makes LLM responses near-instant.

- **Free tier** available at [console.groq.com](https://console.groq.com)
- We use Groq as our LLM provider for two tasks:
  1. **Grading** retrieved chunks (is this chunk relevant?)
  2. **Generating** the final answer

**Why Groq instead of OpenAI?**
- Free tier with generous limits
- Very fast response times
- Hosts open-source models (LLaMA) rather than proprietary ones

**Temperature = 0:**
When we set `temperature=0`, the LLM always picks the most likely next word. This makes responses **deterministic** and **factual** — exactly what we want for a RAG system (no creativity needed, just accuracy).

### 2.7 LangChain — The Glue

LangChain is a **framework** (a library of tools) that makes it easy to build LLM applications. Think of it as a toolkit that provides pre-built components:

**What LangChain gives us in this project:**

| Component | What It Does | LangChain Class |
|-----------|-------------|-----------------|
| PDF Loading | Reads PDF files and extracts text | `PyPDFLoader` |
| Text Splitting | Splits text into chunks | `RecursiveCharacterTextSplitter` |
| Embeddings | Converts text to vectors | `HuggingFaceEmbeddings` |
| Vector Store | Interface to Pinecone | `PineconeVectorStore` |
| LLM | Interface to Groq | `ChatGroq` |
| Documents | Standardized text + metadata container | `Document` |
| Web Search | Interface to Tavily search | `TavilySearchResults` |

**The `Document` object — LangChain's universal data container:**

```python
Document(
    page_content="Backpropagation computes the gradient...",  # The actual text
    metadata={"source": "ml.pdf", "page": 5}                 # Extra info
)
```

Every component in LangChain passes data around as `Document` objects. This standardization is why the library is so useful — everything speaks the same "language."

> [!NOTE]
> **LangChain packages we use:**
> - `langchain` — Core framework (text splitters, document types)
> - `langchain-groq` — Groq LLM integration
> - `langchain-huggingface` — HuggingFace embeddings integration
> - `langchain-pinecone` — Pinecone vector store integration
> - `langchain-community` — Community tools (Tavily)

### 2.8 LangGraph — The Brain (State Machines)

This is probably the most new/complex concept for you. Let's break it down carefully.

**The Problem LangGraph Solves:**
In our CRAG system, we have multiple steps with **conditional logic** (if relevant → do X, if not → do Y). A simple sequential script would work, but becomes messy. LangGraph lets us define this as a **graph** (a flowchart) where:
- **Nodes** = Processing steps (functions)
- **Edges** = Connections between steps
- **Conditional edges** = "If X, go to step A; otherwise go to step B"
- **State** = Data that flows through the graph

**What is a State Machine?**
A state machine is a model where:
1. The system is in one **state** at a time
2. Based on the current state/data, it **transitions** to the next state
3. Each state does some work and updates the data

**Our CRAG Graph visualized:**

```
                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐
                    │ retrieve  │  ← Search ChromaDB for relevant chunks
                    └─────┬─────┘
                          │
                          ▼
                ┌─────────────────┐
                │ grade_documents │  ← LLM checks: are chunks relevant?
                └────────┬────────┘
                         │
                    ┌────┴────┐
                    │ Decision │
                    └────┬────┘
                   /          \
          relevant?            irrelevant?
           (NO web)            (YES web)
              │                    │
              │                    ▼
              │           ┌────────────┐
              │           │ web_search │  ← Tavily search as fallback
              │           └──────┬─────┘
              │                  │
              ▼                  ▼
         ┌──────────────────────────┐
         │        generate         │  ← LLM generates final answer
         └────────────┬────────────┘
                      │
                      ▼
                 ┌─────────┐
                 │   END   │
                 └─────────┘
```

**The State — data that flows through the graph:**

```python
class GraphState(TypedDict):
    question: str                    # The user's question (set at start)
    documents: List[Document]        # Raw retrieved chunks (set by 'retrieve')
    relevant_documents: List[Document]  # Filtered chunks (set by 'grade' or 'web_search')
    used_web_search: bool            # Did we fall back to web? (set by 'grade')
    answer: str                      # The final answer (set by 'generate')
    sources: List[str]               # Source attributions (set by 'generate')
```

**How data flows:**

```
Step 1: retrieve
  Input state:  { question: "What is backprop?" }
  Action:       Search ChromaDB → get 5 chunks
  Output state: { question: "...", documents: [chunk1, chunk2, ...] }

Step 2: grade_documents
  Input state:  { ..., documents: [chunk1, chunk2, ...] }
  Action:       LLM grades each chunk → 3 are relevant, 2 are not
  Output state: { ..., relevant_documents: [chunk1, chunk3, chunk5], used_web_search: False }

Step 3a (if relevant): generate
  Input state:  { ..., relevant_documents: [...], used_web_search: False }
  Action:       LLM generates answer from relevant chunks
  Output state: { ..., answer: "Backprop is...", sources: ["ml.pdf page 5"] }

--- OR ---

Step 3b (if irrelevant): web_search → generate
  Input state:  { ..., used_web_search: True }
  Action:       Tavily searches the web → gets results → LLM generates answer
  Output state: { ..., answer: "Backprop is...", sources: ["https://wikipedia.org/..."] }
```

**Key LangGraph concepts:**
- `StateGraph(GraphState)` — Creates a graph with our state schema
- `graph.add_node("name", function)` — Adds a processing step
- `graph.add_edge("A", "B")` — Connects step A → step B (always)
- `graph.add_conditional_edges("A", condition_fn, mapping)` — Conditional routing
- `graph.compile()` — Compiles the graph into a runnable pipeline
- `compiled_graph.invoke(initial_state)` — Runs the full pipeline

> [!IMPORTANT]
> **Why LangGraph instead of just if/else?**
> 1. **Clarity** — The graph structure makes the logic *visible* and self-documenting
> 2. **Extensibility** — Easy to add new nodes (e.g., a "rewrite query" step) without rewriting everything
> 3. **Debugging** — You can inspect the state at each step
> 4. **Production-ready** — LangGraph handles state management, error handling, and can be scaled
> 
> For this project, simple if/else would work fine. We use LangGraph because it's **industry-standard** for building agentic AI systems and great to learn.

### 2.9 Tavily — Web Search Fallback

Tavily is a **search API designed specifically for AI applications**. When our documents don't have the answer, we fall back to Tavily.

- Free tier at [tavily.com](https://tavily.com) (1,000 searches/month)
- Returns structured results (title, URL, content snippet)
- Optimized for LLM consumption (clean, relevant results)

**How we use it:**
1. The grader decides chunks are irrelevant → triggers web search
2. We send the user's question to Tavily
3. Tavily returns top 3 results with URLs and content
4. We convert these into LangChain `Document` objects
5. The generator uses these web results as context instead of document chunks

---

## 3. Architecture & Data Flow

### The Complete Pipeline

```
┌────────────────────────────── INGESTION PHASE (One-time per upload) ──────────────────────────────┐
│                                                                                                   │
│  PDF Files  ──►  Extract Text  ──►  Split into Chunks  ──►  Generate Embeddings  ──►  ChromaDB   │
│  (upload)       (pypdf)            (500 chars each)         (BGE model, local)       (persistent) │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────── QUERY PHASE (Every question) ──────────────────────────────────────┐
│                                                                                                   │
│  User Question                                                                                    │
│       │                                                                                           │
│       ▼                                                                                           │
│  ┌─────────────┐                                                                                  │
│  │  RETRIEVE   │  Embed the question → search ChromaDB → get top 5 chunks                        │
│  └──────┬──────┘                                                                                  │
│         ▼                                                                                         │
│  ┌─────────────┐                                                                                  │
│  │   GRADE     │  LLM evaluates each chunk: "Is this relevant to the question?"                  │
│  └──────┬──────┘                                                                                  │
│         │                                                                                         │
│    ┌────┴────┐                                                                                    │
│    │ DECIDE  │──── Mostly relevant? ──► Use filtered doc chunks ──────┐                           │
│    └────┬────┘                                                        │                           │
│         │                                                             │                           │
│    Mostly irrelevant?                                                 │                           │
│         │                                                             │                           │
│         ▼                                                             ▼                           │
│  ┌──────────────┐                                              ┌────────────┐                     │
│  │  WEB SEARCH  │  Tavily API → 3 web results                 │  GENERATE  │                     │
│  └──────┬───────┘                                              │            │                     │
│         │                                                      │ LLM builds │                     │
│         └──────────────────────────────────────────────────────►│ answer from│                     │
│                                                                │ context    │                     │
│                                                                └─────┬──────┘                     │
│                                                                      │                            │
│                                                                      ▼                            │
│                                                              Final Answer + Sources               │
│                                                              + "From Docs 📄" or "From Web 🌐"   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### How the Technologies Connect

```
  ┌──────────────────┐         ┌──────────────────┐
  │  HTML/CSS/JS     │  HTTP   │   FastAPI         │
  │  Frontend        │ ◄─────► │   Backend (main.py)│
  │  (static/)       │  REST   │                    │
  └──────────────────┘         └────────┬───────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   LangGraph     │  ← Orchestrator (graph.py)
                               │   (graph.py)    │
                               └────────┬────────┘
                                        │
                  ┌─────────┬──────────┴────────┬──────────┐
                  ▼         ▼                   ▼          ▼
           ┌────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
           │Retrieve│ │  Grade   │  │Web Search│ │ Generate │
           │        │ │          │  │          │ │          │
           │ChromaDB│ │Groq LLM │  │ Tavily   │ │Groq LLM │
           │  +BGE  │ │          │  │   API    │ │          │
           └────────┘ └──────────┘  └──────────┘ └──────────┘
               ▲
               │
        ┌──────────┐
        │ Ingestor │  ← PDF processing (ingestor.py)
        │ pypdf+BGE│
        └──────────┘
```

---

## 4. File-by-File Breakdown

### 4.1 `rag/ingestor.py` — The Document Processor

**Purpose:** Takes uploaded PDFs → extracts text → splits into chunks → embeds → stores in ChromaDB

**Concepts to understand first:**
- PDF text extraction (pypdf reads PDF pages as strings)
- Text splitting (RecursiveCharacterTextSplitter)
- Embeddings (HuggingFaceEmbeddings)
- Pinecone index creation and upserting vectors

**What this file does step by step:**

```
Input:  List of PDF files uploaded via the frontend (sent as multipart/form-data to FastAPI)
Output: Number of chunks stored in Pinecone

Step 1: For each PDF file
        → Read all pages → extract text from each page
        → Create a Document(page_content=text, metadata={source: filename, page: page_num})

Step 2: Pass all Documents to RecursiveCharacterTextSplitter
        → Returns a list of smaller Document chunks (each ~500 chars)

Step 3: Initialize HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        → This downloads the model on first run (~130MB)

Step 4: Connect to Pinecone and upsert chunks
        → PineconeVectorStore.from_documents(documents=chunks, embedding=embedding_model,
              index_name="rag-docs")
        → LangChain embeds each chunk and upserts (insert/update) to Pinecone cloud

Step 5: Return the total count of chunks
```

**Key decisions:**
- `chunk_size=500` — Balanced between too granular and too vague
- `chunk_overlap=100` — Prevents losing context at chunk boundaries
- Pinecone index uses `cosine` similarity metric (best for BGE embeddings)
- Data lives in the cloud — no local storage needed, persists across deploys

---

### 4.2 `rag/retriever.py` — The Search Engine

**Purpose:** Given a question, find the most relevant chunks from ChromaDB

**Concepts to understand first:**
- Similarity search (find vectors closest to the query vector)
- MMR search (Maximal Marginal Relevance — see section 2.4.1)
- Top-K retrieval (return K most similar results)

**What this file does step by step:**

```
Input:  A query string (e.g., "What is backpropagation?")
Output: List of 5 Document objects (most relevant AND diverse chunks)

Step 1: Connect to the existing Pinecone index
        → Same embedding model and index_name as ingestor

Step 2: Convert Pinecone to a LangChain retriever with MMR
        → retriever = vectorstore.as_retriever(
              search_type="mmr",
              search_kwargs={"k": 5, "fetch_k": 20}
          )
        → fetch_k=20: first fetches 20 candidates, then MMR selects the best 5

Step 3: Perform MMR search
        → retriever.invoke(query) → returns top 5 diverse, relevant Documents

Step 4: Return the Documents
```

**Why MMR instead of plain similarity search?**
- Plain search often returns near-duplicate chunks from adjacent paragraphs
- MMR balances **relevance** with **diversity** — each chunk adds new information
- `fetch_k=20` gives MMR a bigger pool to select diverse results from

**Why 5 chunks?**
- Fewer (1-2): Might miss important context
- More (10+): Adds noise, slower grading, may exceed LLM context window
- 5 is a good balance for most use cases

---

### 4.3 `rag/grader.py` — The Quality Filter

**Purpose:** LLM evaluates each retrieved chunk: "Is this relevant to the question?"

**Concepts to understand first:**
- Prompt engineering (crafting specific instructions for the LLM)
- Binary classification (yes/no decisions)
- Threshold-based decisions

**What this file does step by step:**

```
Input:  Question + List of 5 retrieved Documents
Output: Filtered list of relevant Documents + used_web_search flag (bool)

Step 1: Initialize Groq LLM (llama-3.3-70b-versatile, temperature=0)

Step 2: For each document chunk:
        → Send prompt to LLM:
          "You are a relevance grader. Given this chunk and this question,
           respond with only 'yes' or 'no'."
        → Parse response: "yes" → keep chunk, "no" → discard chunk

Step 3: Count relevant vs irrelevant chunks
        → If more than half are irrelevant OR zero pass: set used_web_search = True
        → Otherwise: set used_web_search = False

Step 4: Return filtered_documents and used_web_search flag
```

**Why use an LLM for grading instead of just embedding similarity scores?**
- Embedding similarity is a **coarse** measure — high similarity doesn't always mean relevant
- An LLM can understand **nuance** — e.g., a chunk about "neural network training" has high similarity to "backpropagation" but might not actually discuss backpropagation
- The LLM grader provides a **semantic understanding** of relevance

**Edge case — no documents in ChromaDB:**
If the user hasn't uploaded any documents yet, the retriever returns 0 chunks. In this case, we **skip grading entirely** and go straight to web search.

---

### 4.4 `rag/web_search.py` — The Fallback

**Purpose:** When documents can't answer the question, search the web via Tavily

**Concepts to understand first:**
- API calls to external search services
- Converting search results to a standardized format (LangChain Documents)

**What this file does step by step:**

```
Input:  A query string
Output: List of 3 Document objects (from web results)

Step 1: Initialize TavilySearchResults(max_results=3)

Step 2: Invoke the search with the query
        → Returns raw results: [{url, title, content}, ...]

Step 3: Convert each result to a LangChain Document:
        → Document(
            page_content=result["content"],
            metadata={"source": result["url"], "title": result["title"]}
          )

Step 4: Return the list of Documents
```

**Why only 3 results?**
- More results = more context = slower LLM response
- Top 3 web results are usually sufficient for most questions
- Keeps within free tier limits

---

### 4.5 `rag/generator.py` — The Answer Builder

**Purpose:** Takes context (from docs or web) + question → generates a final answer

**Concepts to understand first:**
- Prompt construction (assembling context + question into a structured prompt)
- Temperature settings (0 = deterministic/factual)
- Source attribution (tracking where the answer came from)

**What this file does step by step:**

```
Input:  Question + context Documents + used_web_search flag
Output: Answer string + list of source strings

Step 1: Format the context documents into a readable string:
        → "Source: ml.pdf (page 5)\nContent: Backpropagation computes..."
        → "Source: ml.pdf (page 6)\nContent: The gradient is then..."

Step 2: Build the full prompt:
        → System instruction + formatted context + question

Step 3: Send to Groq LLM (temperature=0)
        → Get the generated answer

Step 4: Extract unique sources from document metadata
        → If from docs: ["ml.pdf page 5", "ml.pdf page 6"]
        → If from web: ["https://en.wikipedia.org/...", "https://..."]

Step 5: Return answer + sources
```

---

### 4.6 `rag/graph.py` — The Orchestrator

**Purpose:** Connects all the above components into a single LangGraph pipeline

**Concepts to understand first:**
- LangGraph StateGraph (see section 2.8 above)
- TypedDict (Python typed dictionaries for state schema)
- Node functions (each node reads state, does work, returns updated state)
- Conditional edges (routing based on state values)

**What this file does step by step:**

```
Step 1: Define GraphState (TypedDict with all fields)

Step 2: Create node functions:
        - retrieve_node(state) → calls retriever.py → returns {"documents": [...]}
        - grade_node(state) → calls grader.py → returns {"relevant_documents": [...], "used_web_search": bool}
        - web_search_node(state) → calls web_search.py → returns {"relevant_documents": [...]}
        - generate_node(state) → calls generator.py → returns {"answer": "...", "sources": [...]}

Step 3: Create conditional router:
        - decide_search(state): 
          if state["used_web_search"] == True → return "web_search"
          else → return "generate"

Step 4: Build the graph:
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade_documents", grade_node)
        graph.add_node("web_search", web_search_node)
        graph.add_node("generate", generate_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            decide_search,
            {"web_search": "web_search", "generate": "generate"}
        )
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)

Step 5: Compile and expose:
        compiled = graph.compile()

        def run_graph(question: str) -> dict:
            result = compiled.invoke({"question": question})
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "used_web_search": result["used_web_search"]
            }
```

> [!NOTE]
> **How `add_conditional_edges` works:**
> - First arg: The source node (`"grade_documents"`)
> - Second arg: A function that returns a string (`decide_search`)
> - Third arg: A mapping of string → node name (`{"web_search": "web_search", "generate": "generate"}`)
> 
> The function examines the state and returns a key. LangGraph uses that key to look up which node to go to next.

---

### 4.7 `main.py` — FastAPI Backend

**Purpose:** REST API that the frontend calls to upload docs and ask questions

**Concepts to understand first:**
- FastAPI basics (routes, request/response, async)
- REST API design (endpoints, HTTP methods, JSON)
- CORS (allowing frontend to talk to backend)
- File uploads (multipart/form-data)

**What is FastAPI?**
FastAPI is a modern Python web framework for building APIs. Unlike Streamlit (which bundles UI + logic), FastAPI is **backend only** — it receives HTTP requests and returns JSON responses. A separate HTML/CSS/JS frontend sends requests to it.

**Key FastAPI concepts:**

| Concept | Explanation |
|---------|-------------|
| `@app.post("/upload")` | Defines a POST endpoint at `/upload` |
| `@app.post("/query")` | Defines a POST endpoint at `/query` |
| `@app.get("/status")` | Defines a GET endpoint at `/status` |
| `UploadFile` | FastAPI's type for handling file uploads |
| `JSONResponse` | Returns a JSON response to the client |
| `CORSMiddleware` | Allows the frontend (different origin) to call the API |
| `StaticFiles` | Serves the HTML/CSS/JS frontend files |

**API Endpoints we’ll build:**

```
POST /api/upload     → Accepts PDF files, processes & stores in Pinecone
                       Returns: {"chunk_count": 47, "message": "Success"}

POST /api/query      → Accepts {"question": "What is backprop?"}
                       Runs the LangGraph pipeline
                       Returns: {"answer": "...", "sources": [...], "used_web_search": false}

GET  /api/status     → Returns whether documents are loaded
                       Returns: {"docs_loaded": true, "chunk_count": 47}

DELETE /api/clear    → Deletes all vectors from Pinecone index
                       Returns: {"message": "Database cleared"}

GET  /               → Serves the HTML frontend (static files)
```

> [!NOTE]
> **FastAPI vs Streamlit — why the switch?**
> - **Separation of concerns**: Backend logic is cleanly separated from frontend UI
> - **Standard API**: Any frontend (React, mobile app, etc.) can consume the API
> - **Render deployment**: FastAPI deploys easily on Render’s free tier with `uvicorn`
> - **Production-ready**: FastAPI is used in real production systems (Streamlit is more for prototyping)

---

### 4.8 `static/` — HTML/CSS/JS Frontend

**Purpose:** A beautiful chat interface served by FastAPI as static files

The frontend is a single-page app with:
- **Sidebar**: PDF upload area, document status, clear button
- **Main area**: Chat interface with messages, source badges, expandable sources
- Uses `fetch()` API to call the FastAPI endpoints

```
static/
├── index.html    → Page structure
├── style.css     → Styling (dark theme, glassmorphism, animations)
└── script.js     → Logic (API calls, chat rendering, file upload)
```

**How frontend ↔ backend communicate:**

```
User clicks "Upload PDFs"
  → script.js sends POST /api/upload with FormData (files)
  → FastAPI processes files, returns {chunk_count: 47}
  → script.js updates the status badge

User types a question and hits Enter
  → script.js sends POST /api/query with {question: "..."}
  → FastAPI runs the CRAG pipeline, returns {answer, sources, used_web_search}
  → script.js renders the response with 📄 or 🌐 badge
```

---

## 5. Implementation Order & Learning Path

> [!IMPORTANT]
> Follow this order. Each step builds on the previous one. Don't skip ahead.

### Phase 1: Setup & Foundation (30 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 1.1 | Create project directory structure | Project organization |
| 1.2 | Create `requirements.txt` | Python dependency management |
| 1.3 | Create virtual environment and install dependencies | `venv`, `pip install` |
| 1.4 | Create `.env.example` and `.env` with your API keys | Environment variables, secrets management |
| 1.5 | Get your free API keys from Groq, Tavily, and Pinecone | API authentication |

**After Phase 1, you should be able to:**
- Run `python -c "import langchain; print(langchain.__version__)"` without errors
- Have both API keys ready

---

### Phase 2: Ingestion Pipeline (45 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 2.1 | Build `rag/ingestor.py` | PDF parsing, text splitting, embeddings, Pinecone |
| 2.2 | Test with a sample PDF | Verify chunks are stored correctly |

**Test checkpoint:**
```python
# Quick test script
from rag.ingestor import ingest_pdfs
count = ingest_pdfs(["sample.pdf"])
print(f"Stored {count} chunks")  # Should print a number > 0
```

**After Phase 2, you should be able to:**
- Take a PDF → split into chunks → embed → store in Pinecone
- See vectors in your Pinecone dashboard at [app.pinecone.io](https://app.pinecone.io)

---

### Phase 3: Retrieval (20 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 3.1 | Build `rag/retriever.py` | Similarity search, top-K retrieval |
| 3.2 | Test with a sample query | Verify relevant chunks are returned |

**Test checkpoint:**
```python
from rag.retriever import retrieve_documents
docs = retrieve_documents("What is machine learning?")
for doc in docs:
    print(doc.page_content[:100], "...\n")  # Should show relevant chunks
```

---

### Phase 4: Grading (30 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 4.1 | Build `rag/grader.py` | LLM prompting, binary classification |
| 4.2 | Test with relevant and irrelevant queries | Verify grading works correctly |

**Test checkpoint:**
```python
from rag.retriever import retrieve_documents
from rag.grader import grade_documents

# Test with a relevant query (should NOT trigger web search)
docs = retrieve_documents("your PDF topic here")
relevant, web_search = grade_documents("your PDF topic here", docs)
print(f"Relevant: {len(relevant)}, Web search: {web_search}")

# Test with an irrelevant query (SHOULD trigger web search)
docs = retrieve_documents("Who won the 2024 World Cup?")
relevant, web_search = grade_documents("Who won the 2024 World Cup?", docs)
print(f"Relevant: {len(relevant)}, Web search: {web_search}")  # web_search should be True
```

---

### Phase 5: Web Search (15 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 5.1 | Build `rag/web_search.py` | External API integration, data transformation |
| 5.2 | Test with a general question | Verify web results are returned |

**Test checkpoint:**
```python
from rag.web_search import search_web
results = search_web("What is the capital of France?")
for doc in results:
    print(doc.metadata["source"], "-", doc.page_content[:100])
```

---

### Phase 6: Generation (20 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 6.1 | Build `rag/generator.py` | Prompt engineering, LLM response generation |
| 6.2 | Test with both doc-based and web-based context | Verify answers are accurate |

**Test checkpoint:**
```python
from rag.generator import generate_answer

# Test with fake documents
from langchain.schema import Document
docs = [Document(page_content="Paris is the capital of France.", metadata={"source": "geo.pdf"})]
answer, sources = generate_answer("What is the capital of France?", docs, used_web_search=False)
print(answer)
print(sources)
```

---

### Phase 7: LangGraph Pipeline (45 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 7.1 | Build `rag/graph.py` | State machines, graph construction, conditional routing |
| 7.2 | Test the full pipeline end-to-end | Integration testing |

**Test checkpoint:**
```python
from rag.graph import run_graph

# Test with a relevant query (should use docs)
result = run_graph("What is [topic from your PDF]?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Used web: {result['used_web_search']}")  # Should be False

# Test with an irrelevant query (should use web)
result = run_graph("What's the latest news about SpaceX?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Used web: {result['used_web_search']}")  # Should be True
```

> [!TIP]
> This is the most rewarding phase — you'll see all the pieces working together for the first time!

---

### Phase 8: FastAPI Backend + Frontend (60 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 8.1 | Build `main.py` — FastAPI app with `/api/upload` endpoint | FastAPI basics, file uploads |
| 8.2 | Add `/api/query` endpoint integrated with LangGraph | REST API + pipeline integration |
| 8.3 | Add `/api/status` and `/api/clear` endpoints | CRUD operations |
| 8.4 | Build `static/index.html` — chat UI with sidebar | HTML structure |
| 8.5 | Build `static/style.css` — dark theme, glassmorphism | CSS design |
| 8.6 | Build `static/script.js` — API calls, chat rendering | JavaScript, fetch API |
| 8.7 | Add source badges (📄/🌐) and expandable sources | UI polish |
| 8.8 | Add error handling and loading states | User experience |

**Launch the app:**
```bash
uvicorn main:app --reload
# Open http://localhost:8000 in your browser
```

---

### Phase 9: Documentation & Deployment (45 minutes)

| Step | Task | What You'll Learn |
|------|------|-------------------|
| 9.1 | Write `README.md` | Technical documentation |
| 9.2 | Create `render.yaml` (optional) or configure via dashboard | Infrastructure as code |
| 9.3 | Deploy to Render | Cloud deployment, environment variables |

**Render deployment steps:**
1. Push project to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables in Render dashboard:
   - `GROQ_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`
7. Deploy — live URL in ~3 minutes

> [!TIP]
> Since Pinecone is cloud-hosted, your document data **persists across deploys** — no re-upload needed!

---

## 6. Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — enhancing LLM responses with retrieved context |
| **CRAG** | Corrective RAG — adds a grading step to verify chunk relevance before generating |
| **Embedding** | A numerical vector representation of text that captures semantic meaning |
| **Vector Database** | A database optimized for storing and searching embedding vectors |
| **Pinecone** | A cloud-hosted, managed vector database we use for storing document embeddings |
| **Index** | A Pinecone container for vectors (similar to a database table) |
| **Namespace** | A partition within a Pinecone index for organizing data |
| **MMR** | Maximal Marginal Relevance — a search method that balances relevance with diversity |
| **Chunk** | A small segment of text split from a larger document |
| **Similarity Search** | Finding vectors closest to a query vector (semantically similar text) |
| **LLM** | Large Language Model — a neural network that generates human-like text |
| **Groq** | A fast LLM inference provider (runs LLaMA models) |
| **LangChain** | A Python framework providing tools for building LLM applications |
| **LangGraph** | A LangChain extension for building stateful, graph-based AI workflows |
| **StateGraph** | A LangGraph construct defining nodes (steps), edges (connections), and state |
| **Node** | A processing step in a LangGraph graph |
| **Edge** | A connection between two nodes in a graph |
| **Conditional Edge** | An edge that routes to different nodes based on a condition |
| **Tavily** | A search API optimized for AI applications (our web search fallback) |
| **FastAPI** | A modern Python web framework for building REST APIs |
| **REST API** | An interface where clients send HTTP requests and receive JSON responses |
| **CORS** | Cross-Origin Resource Sharing — allows a frontend to call a backend on a different origin |
| **Uvicorn** | An ASGI server that runs FastAPI applications |
| **Render** | A cloud platform for deploying web applications (free tier available) |
| **Temperature** | An LLM parameter: 0 = deterministic/factual, 1 = creative/random |
| **Cosine Similarity** | A measure of angle between two vectors — used to find similar embeddings |
| **Top-K** | Returning the K most similar results from a search |
| **Prompt Engineering** | Crafting input text to get optimal LLM responses |
| **Ingestion** | The process of loading, processing, and storing documents |
| **Hallucination** | When an LLM generates plausible-sounding but factually incorrect information |
| **Context Window** | The maximum number of tokens an LLM can process at once |

---

## 7. Common Pitfalls & Debugging Tips

### Pitfall 1: Pinecone Index Not Found
**Cause:** Trying to query an index that hasn't been created yet.
**Fix:** The ingestor should create the index if it doesn't exist using `pc.create_index()`. Check your Pinecone dashboard to verify.

### Pitfall 2: Pinecone Dimension Mismatch
**Cause:** The index was created with a different embedding dimension than your model outputs.
**Fix:** BGE-small produces 384-dimensional vectors. Make sure your index is created with `dimension=384`. If mismatched, delete and recreate the index.

### Pitfall 3: Embedding Model Downloads Every Time
**Cause:** The model isn't being cached.
**Fix:** HuggingFace caches models in `~/.cache/huggingface/`. The first run downloads ~130MB, subsequent runs use the cache. On Render, the cache persists within a deploy.

### Pitfall 4: Groq API Rate Limits
**Cause:** Free tier has rate limits (30 requests/min for some models).
**Fix:** Add `time.sleep(1)` between grading calls if hitting limits. Or batch grading into fewer calls.

### Pitfall 5: CORS Errors in Browser Console
**Cause:** Frontend can't call backend because CORS isn't configured.
**Fix:** Add `CORSMiddleware` to your FastAPI app with `allow_origins=["*"]` during development.

### Pitfall 6: API Keys Not Found
**Cause:** `.env` file not in the right location or `load_dotenv()` not called.
**Fix:** Use `os.getenv("GROQ_API_KEY")` and set environment variables in Render's dashboard for production.

### Pitfall 7: Grader Returns Unexpected Responses
**Cause:** LLM might return "Yes" instead of "yes", or "Yes, this is relevant."
**Fix:** Parse with `.strip().lower().startswith("yes")` instead of exact match.

### Pitfall 8: PDF Text Extraction Returns Empty
**Cause:** Some PDFs are image-based (scanned documents) — pypdf can't extract text from images.
**Fix:** Add a check — if extracted text is empty, inform the user that the PDF might be a scanned image and needs OCR processing (out of scope for this project).

### Pitfall 9: Render Deploy Fails on Large Dependencies
**Cause:** `sentence-transformers` + `torch` can be large. Render free tier has 512MB RAM.
**Fix:** Use `--no-cache-dir` in pip install. If RAM is tight, consider using Render's starter plan or a lighter embedding model.

---

## 8. Resources & Further Reading

### Must-Read Before Coding

| Resource | What You'll Learn | Time |
|----------|-------------------|------|
| [LangChain Docs — Quickstart](https://python.langchain.com/docs/get_started/quickstart) | Core LangChain concepts | 20 min |
| [LangGraph Docs — Introduction](https://langchain-ai.github.io/langgraph/) | What LangGraph is and how it works | 20 min |
| [Pinecone Quickstart](https://docs.pinecone.io/guides/get-started/quickstart) | How to use Pinecone | 15 min |
| [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) | FastAPI basics, routes, requests | 30 min |

### Video Tutorials (Recommended)

| Video | Topic |
|-------|-------|
| Search YouTube: "Corrective RAG LangGraph" | End-to-end CRAG implementation |
| Search YouTube: "LangGraph Tutorial for Beginners" | LangGraph fundamentals |
| Search YouTube: "FastAPI Full Course" | FastAPI from scratch |
| Search YouTube: "Pinecone Vector Database Tutorial" | Pinecone basics |

### Original Paper

| Paper | What It Covers |
|-------|---------------|
| [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) | The academic paper that proposed CRAG |

---

## Ready to Start?

Once you've read through this guide and feel comfortable with the concepts, let me know and we'll start coding file by file, following the implementation order in Phase 1-9 above.

**Before we begin, make sure you have:**
- [ ] Python 3.9+ installed
- [ ] A Groq API key (free at [console.groq.com](https://console.groq.com))
- [ ] A Tavily API key (free at [tavily.com](https://tavily.com))
- [ ] A Pinecone API key (free at [pinecone.io](https://www.pinecone.io/))
- [ ] Read through the LangGraph introduction (link above)
- [ ] Read through the FastAPI tutorial (link above)
- [ ] A PDF file ready for testing (any topic — your course notes work great!)
