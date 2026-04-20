"""
Graph Module — LangGraph CRAG State Machine

This is the orchestrator that connects all RAG components into a single
executable pipeline using LangGraph's StateGraph.

The flow:
    START → retrieve → grade_documents → [conditional]
                                           ├── relevant → generate → END
                                           └── irrelevant → web_search → generate → END

Each node is a function that reads from the shared state, does its work,
and returns a dict of state fields to update.
"""

from typing import TypedDict, List

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from rag.retriever import retrieve_documents
from rag.grader import grade_documents
from rag.web_search import search_web
from rag.generator import generate_answer


# ---------------------------------------------------------------------------
# State schema — the data structure that flows through the graph
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str                       # The user's question (set at start)
    documents: List[Document]           # Raw retrieved chunks (set by 'retrieve')
    relevant_documents: List[Document]  # Filtered chunks (set by 'grade' or 'web_search')
    used_web_search: bool               # Flag: did we fall back to web? (set by 'grade')
    answer: str                         # The final answer (set by 'generate')
    sources: List[str]                  # Source attributions (set by 'generate')


# ---------------------------------------------------------------------------
# Node functions — each one is a step in the pipeline
# ---------------------------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    """
    Node 1: Retrieve relevant chunks from Pinecone.

    Reads:  state["question"]
    Sets:   state["documents"]
    """
    question = state["question"]
    documents = retrieve_documents(question)
    return {"documents": documents}


def grade_documents_node(state: GraphState) -> dict:
    """
    Node 2: Grade each retrieved chunk for relevance using the LLM.

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
    Node 3 (conditional): Search the web when document chunks are irrelevant.

    Reads:  state["question"]
    Sets:   state["relevant_documents"]
    """
    question = state["question"]
    web_docs = search_web(question)
    return {"relevant_documents": web_docs}


def generate_node(state: GraphState) -> dict:
    """
    Node 4: Generate the final answer from the context documents.

    Reads:  state["question"], state["relevant_documents"], state["used_web_search"]
    Sets:   state["answer"], state["sources"]
    """
    question = state["question"]
    relevant_docs = state["relevant_documents"]
    used_web_search = state["used_web_search"]
    answer, sources = generate_answer(question, relevant_docs, used_web_search)
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Conditional edge — decides whether to web search or generate directly
# ---------------------------------------------------------------------------
def decide_search(state: GraphState) -> str:
    """
    Conditional router: after grading, decide the next step.

    If used_web_search is True  → route to "web_search"
    If used_web_search is False → route to "generate"
    """
    if state["used_web_search"]:
        return "web_search"
    return "generate"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_node)

# Add edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
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
def run_graph(question: str) -> dict:
    """
    Run the full Corrective RAG pipeline.

    Args:
        question: The user's question string.

    Returns:
        A dict with:
          - "answer": The generated answer string
          - "sources": List of source identifiers (filenames or URLs)
          - "used_web_search": Whether the answer came from web search
    """
    result = compiled_graph.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "used_web_search": result["used_web_search"],
    }
