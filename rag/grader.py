"""
Grader Module — LLM-Based Chunk Relevance Grading

This module is the "corrective" part of Corrective RAG.
It uses Groq's LLM to evaluate each retrieved chunk:
    "Is this chunk actually relevant to the user's question?"

If more than half the chunks are irrelevant (or none are found),
the system falls back to web search instead of generating a
potentially hallucinated answer.
"""

from typing import List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag import GOOGLE_API_KEY, LLM_MODEL


# The grading prompt — asks the LLM for a simple yes/no relevance judgment
GRADING_PROMPT_TEMPLATE = """You are a relevance grader. Given the following retrieved document chunk and user question, \
respond with only "yes" if the chunk is relevant to answering the question, or "no" if it is not.

Question: {question}

Document chunk: {chunk_content}

Respond with only: yes or no"""


def grade_documents(
    question: str,
    documents: List[Document],
) -> Tuple[List[Document], bool]:
    """
    Grade each retrieved document chunk for relevance to the question.

    Args:
        question:  The user's question string.
        documents: List of retrieved Document objects from the retriever.

    Returns:
        A tuple of:
          - relevant_docs: List of Documents that passed the relevance check.
          - used_web_search: True if we should fall back to web search.

    Decision logic:
        - If no documents were retrieved → web search (skip grading entirely)
        - If >50% of chunks are irrelevant → web search
        - If 0 chunks pass grading → web search
        - Otherwise → use the relevant chunks for generation
    """

    # Edge case: no documents to grade (empty Pinecone or no docs uploaded)
    if not documents:
        return [], True

    # Initialize Gemini LLM with temperature=0 for deterministic grading
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    relevant_docs: List[Document] = []

    for doc in documents:
        prompt = GRADING_PROMPT_TEMPLATE.format(
            question=question,
            chunk_content=doc.page_content,
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            grade = response.content.strip().lower()

            # Use startswith("yes") to handle "Yes", "yes.", "Yes, it is relevant" etc.
            if grade.startswith("yes"):
                relevant_docs.append(doc)

        except Exception as e:
            # On grading error, keep the chunk (fail-open) to avoid losing data
            print(f"[Grader] Error grading chunk: {e}")
            relevant_docs.append(doc)

    # Decision: trigger web search if too many chunks are irrelevant
    total = len(documents)
    relevant_count = len(relevant_docs)
    used_web_search = (relevant_count == 0) or (relevant_count < total / 2)

    return relevant_docs, used_web_search
