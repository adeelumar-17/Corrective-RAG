"""
Graph Module — LangGraph CRAG State Machine

This is the orchestrator that connects all RAG components into a single
executable pipeline using LangGraph's StateGraph.

The flow:
    START → retrieve → classify_question → grade_documents → [conditional]
                                                               ├── relevant → generate → END
                                                               ├── document_meta (always) → generate → END
                                                               └── irrelevant (knowledge) → web_search → generate → END

Each node is a function that reads from the shared state, does its work,
and returns a dict of state fields to update.
"""

from typing import TypedDict, List

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from rag.retriever import retrieve_documents
from rag.grader import grade_documents
from rag.question_classifier import classify_question
from rag.web_search import search_web
from rag.generator import generate_answer


# ---------------------------------------------------------------------------
# State schema — the data structure that flows through the graph
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str                       # The user's question (set at start)
    session_id: str                     # Session ID for per-user isolation
    documents: List[Document]           # Raw retrieved chunks (set by 'retrieve')
    relevant_documents: List[Document]  # Filtered chunks (set by 'grade' or 'web_search')
    question_type: str                  # "document_meta" or "knowledge_query" (set by 'classify')
    used_web_search: bool               # Flag: did we fall back to web? (set by 'grade')
    answer: str                         # The final answer (set by 'generate')
    sources: List[str]                  # Source attributions (set by 'generate')


# ---------------------------------------------------------------------------
# Node functions — each one is a step in the pipeline
# ---------------------------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    """
    Node 1: Retrieve relevant chunks from Pinecone.

    Reads:  state["question"], state["session_id"]
    Sets:   state["documents"]
    """
    question = state["question"]
    session_id = state["session_id"]
    documents = retrieve_documents(question, session_id)
    return {"documents": documents}


def classify_question_node(state: GraphState) -> dict:
    """
    Node 2: Classify the question as document-meta or knowledge query.

    Reads:  state["question"]
    Sets:   state["question_type"]
    """
    question = state["question"]
    question_type = classify_question(question)
    return {"question_type": question_type}


def grade_documents_node(state: GraphState) -> dict:
    """
    Node 3: Grade each retrieved chunk for relevance using the LLM.

    Reads:  state["question"], state["documents"]
    Sets:   state["relevant_documents"], state["used_web_search"]
    """
    question = state["question"]
    documents = state["documents"]
    relevant_docs, used_web_search = grade_documents(question, documents)
    return {
        "relevant_documents": relevant_docs,
        "used_web_search": used_web_search,
    }


def web_search_node(state: GraphState) -> dict:
    """
    Node 4 (conditional): Search the web when document chunks are irrelevant.

    Reads:  state["question"]
    Sets:   state["relevant_documents"]
    """
    question = state["question"]
    web_docs = search_web(question)
    return {"relevant_documents": web_docs}


def generate_node(state: GraphState) -> dict:
    """
    Node 5: Generate the final answer from the context documents.

    For document_meta questions with no relevant docs, uses ALL retrieved
    documents as context (since individual chunk grading is unreliable
    for broad meta-questions).

    Reads:  state["question"], state["relevant_documents"], state["used_web_search"],
            state["question_type"], state["documents"]
    Sets:   state["answer"], state["sources"]
    """
    question = state["question"]
    relevant_docs = state["relevant_documents"]
    used_web_search = state["used_web_search"]
    question_type = state.get("question_type", "knowledge_query")

    # For document_meta questions, if grading filtered out all docs,
    # fall back to using ALL retrieved chunks (they're all from the document)
    if question_type == "document_meta" and not relevant_docs:
        relevant_docs = state.get("documents", [])

    answer, sources = generate_answer(question, relevant_docs, used_web_search)
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Conditional edge — decides whether to web search or generate directly
# ---------------------------------------------------------------------------
def decide_search(state: GraphState) -> str:
    """
    Conditional router: after grading, decide the next step.

    - If question is about the document itself → ALWAYS generate (never web search)
    - If used_web_search is True and it's a knowledge query → route to "web_search"
    - Otherwise → route to "generate"
    """
    question_type = state.get("question_type", "knowledge_query")

    # Document-meta questions should NEVER fall back to web search
    if question_type == "document_meta":
        return "generate"

    if state["used_web_search"]:
        return "web_search"
    return "generate"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("classify_question", classify_question_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_node)

# Add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "classify_question")
graph.add_edge("classify_question", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_search,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

# Compile into a runnable pipeline
compiled_graph = graph.compile()


# ---------------------------------------------------------------------------
# Public API — single entry point for the entire CRAG pipeline
# ---------------------------------------------------------------------------
def run_graph(question: str, session_id: str) -> dict:
    """
    Run the full Corrective RAG pipeline.

    Args:
        question:   The user's question string.
        session_id: Unique session identifier for per-user data isolation.

    Returns:
        A dict with:
          - "answer": The generated answer string
          - "sources": List of source identifiers (filenames or URLs)
          - "used_web_search": Whether the answer came from web search
    """
    result = compiled_graph.invoke({
        "question": question,
        "session_id": session_id,
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "used_web_search": result["used_web_search"],
    }
