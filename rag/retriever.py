"""
Retriever Module — Pinecone Semantic Search with MMR

This module queries Pinecone using Maximal Marginal Relevance (MMR)
to find the most relevant AND diverse document chunks for a given query.

Why MMR over plain similarity?
  Plain similarity often returns near-duplicate chunks from adjacent paragraphs.
  MMR balances relevance with diversity — each returned chunk adds new information.
"""

from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from rag import PINECONE_INDEX_NAME, RETRIEVAL_K, RETRIEVAL_FETCH_K
from rag.ingestor import get_embeddings


def retrieve_documents(query: str) -> List[Document]:
    """
    Retrieve the top-K most relevant (and diverse) document chunks from Pinecone.

    Args:
        query: The user's question string.

    Returns:
        A list of LangChain Document objects, each containing:
          - page_content: the chunk text
          - metadata: {"source": filename, "page": page_number}

    Search strategy:
        Uses MMR with fetch_k=20 candidates, then selects the best k=5.
        This ensures the returned chunks cover different aspects of the topic.
    """
    try:
        embeddings = get_embeddings()

        # Connect to the existing Pinecone index
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

        # Create an MMR retriever
        # - search_type="mmr": Maximal Marginal Relevance
        # - k=5: return 5 final chunks
        # - fetch_k=20: fetch 20 candidates first, then MMR picks the best 5
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_FETCH_K,
            },
        )

        # Perform the search
        docs = retriever.invoke(query)
        return docs

    except Exception as e:
        # If Pinecone index doesn't exist yet (no docs uploaded), return empty
        print(f"[Retriever] Error: {e}")
        return []
