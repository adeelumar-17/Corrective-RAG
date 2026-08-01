"""
Question Classifier Module — Document-Meta vs Knowledge Query Detection

This module classifies whether a user's question is asking about the
uploaded document itself (meta-questions) or asking for external knowledge.

Meta-questions like "What is this document about?" or "Summarize the document"
should NEVER trigger a web search fallback — the answer must come from the
uploaded document chunks, even if the grader considers individual chunks
"irrelevant" (which happens because no single chunk matches a broad question).

Knowledge queries like "What is quantum computing?" can legitimately fall
back to web search if the document doesn't cover the topic.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from rag import GROQ_API_KEY, LLM_MODEL


# Classification prompt — lightweight, single-token response
CLASSIFICATION_PROMPT_TEMPLATE = """You are a question classifier. Determine whether the user's question is:

1. **document_meta** — The question asks about the uploaded document itself.
   Examples: "What is this document about?", "Summarize the document", "Who is the author?",
   "What topics does this cover?", "Give me an overview", "What are the main points?",
   "How long is this document?", "What is discussed in chapter 3?"

2. **knowledge_query** — The question asks for factual or external knowledge that may or may not
   be in the document. These are questions that COULD be answered by a web search if the
   document doesn't contain the answer.
   Examples: "What is machine learning?", "How does photosynthesis work?",
   "What is the capital of France?", "Explain the theory of relativity"

Question: {question}

Respond with ONLY one of: document_meta OR knowledge_query"""


def classify_question(question: str) -> str:
    """
    Classify whether a question is about the document itself or external knowledge.

    Args:
        question: The user's question string.

    Returns:
        "document_meta" if the question is about the document itself.
        "knowledge_query" if the question seeks external/factual knowledge.
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0,
    )

    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(question=question)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        classification = response.content.strip().lower()

        # Normalize the response — be lenient with LLM output
        if "document_meta" in classification:
            return "document_meta"
        elif "knowledge_query" in classification:
            return "knowledge_query"
        else:
            # Default to knowledge_query (safer — allows web search fallback)
            return "knowledge_query"

    except Exception as e:
        print(f"[Classifier] Error classifying question: {e}")
        # On error, default to knowledge_query (fail-open for web search)
        return "knowledge_query"
