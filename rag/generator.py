"""
Generator Module — Final Answer Generation using Groq LLM

This module takes the context documents (from either Pinecone or web search)
and the user's question, then generates a final answer using Groq's LLM.

The answer includes source attribution so the user knows where
the information came from.
"""

from typing import List, Tuple

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag import GROQ_API_KEY, LLM_MODEL


# Generation prompt — instructs the LLM to answer from the provided context
GENERATION_PROMPT_TEMPLATE = """You are an expert research assistant. Your task is to provide a thorough, \
well-structured answer to the user's question based on the provided context.

## Instructions
1. **Use the context below as your primary source.** Synthesize information across multiple sources when relevant.
2. **Cite your sources inline** using the format [Source N] so the user can trace each claim.
3. **Structure your answer clearly:**
   - Use **bold** for key terms and concepts.
   - Use bullet points or numbered lists for multi-part answers.
   - Provide examples or explanations when they add clarity.
4. **If the context is insufficient**, provide what you can from the available sources and clearly state what information is missing or incomplete. Do NOT fabricate information.
5. **If the question is about the document itself** (e.g., "What is this document about?", "Summarize this"), treat ALL provided chunks as relevant context and synthesize a comprehensive answer.
6. **Tone**: Be informative, precise, and helpful. Write for a knowledgeable reader who values depth.

## Context
{formatted_chunks}

## Question
{question}

## Answer"""


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

    # Initialize Groq LLM with temperature=0 for factual, deterministic answers
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
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
