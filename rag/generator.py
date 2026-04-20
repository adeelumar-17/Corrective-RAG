"""
Generator Module — Final Answer Generation using Groq LLM

This module takes the context documents (from either Pinecone or web search)
and the user's question, then generates a final answer using Groq's LLM.

The answer includes source attribution so the user knows where
the information came from.
"""

from typing import List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document, HumanMessage

from rag import GOOGLE_API_KEY, LLM_MODEL


# Generation prompt — instructs the LLM to answer from context only
GENERATION_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using ONLY the context provided below.
Be concise, accurate, and cite which source your answer came from.
If the context does not contain enough information, say so honestly.

Context:
{formatted_chunks}

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    context_docs: List[Document],
    used_web_search: bool,
) -> Tuple[str, List[str]]:
    """
    Generate a final answer from the context documents.

    Args:
        question:       The user's question string.
        context_docs:   List of Document objects (from Pinecone or web search).
        used_web_search: Whether the context came from web search (for logging).

    Returns:
        A tuple of:
          - answer: The generated answer string.
          - sources: List of unique source identifiers (filenames or URLs).
    """

    # Initialize Gemini LLM with temperature=0 for factual, deterministic answers
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    # ---- Format context documents into a readable string ----
    formatted_chunks = ""
    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        title = doc.metadata.get("title", "")

        # Build a human-readable source label
        if page:
            source_label = f"{source} (page {page})"
        elif title:
            source_label = f"{title} — {source}"
        else:
            source_label = source

        formatted_chunks += f"\n[Source {i}: {source_label}]\n{doc.page_content}\n"

    # ---- Build and send the prompt ----
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        formatted_chunks=formatted_chunks,
        question=question,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        answer = f"I'm sorry, I encountered an error generating the answer: {str(e)}"

    # ---- Extract unique source identifiers ----
    sources: List[str] = []
    seen: set = set()
    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        title = doc.metadata.get("title", "")

        if page:
            source_key = f"{source} (page {page})"
        elif title:
            source_key = f"{source}"
        else:
            source_key = source

        if source_key not in seen:
            sources.append(source_key)
            seen.add(source_key)

    return answer, sources
