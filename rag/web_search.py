"""
Web Search Module — Tavily Fallback Search

When the grader determines that the retrieved document chunks are
irrelevant to the user's question, this module searches the web
via Tavily API and returns results as LangChain Documents.

Tavily is a search API optimized for AI applications — it returns
clean, structured results that work well as LLM context.
"""

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document


def search_web(query: str) -> List[Document]:
    """
    Search the web using Tavily and return results as LangChain Documents.

    Args:
        query: The user's question string.

    Returns:
        A list of up to 3 Document objects, each containing:
          - page_content: the search result content/snippet
          - metadata: {"source": URL, "title": result title}

    Note:
        TAVILY_API_KEY must be set as an environment variable.
        This is handled in rag/__init__.py during import.
    """
    try:
        # Initialize Tavily search tool (reads API key from env automatically)
        search = TavilySearchResults(max_results=3)

        # Execute the search
        results = search.invoke(query)

        # Convert raw results to LangChain Document objects
        documents: List[Document] = []
        for result in results:
            doc = Document(
                page_content=result.get("content", "No content available"),
                metadata={
                    "source": result.get("url", "Unknown URL"),
                    "title": result.get("title", "Web Result"),
                },
            )
            documents.append(doc)

        return documents

    except Exception as e:
        print(f"[Web Search] Error: {e}")
        return []
