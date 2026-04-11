"""
Retrieval strategy helpers for LangChain RAG pipelines.

Public API:
    create_retriever()                — vector-store similarity / MMR retriever
    create_self_query_retriever()     — metadata-filtered retriever via LLM
    create_BM25retriever_from_docs()  — sparse BM25 from Documents
    create_BM25retriever_from_text()  — sparse BM25 from plain strings
    create_hybrid_retriever()         — EnsembleRetriever (sparse + semantic)
    reorder_docs()                    — LongContextReorder for retrieved docs
"""

import logging

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)


def create_self_query_retriever(
    model: BaseChatModel,
    vector_store: VectorStore,
    document_content_description: str,
    metadata_field_info: list[AttributeInfo],
    verbose: bool = False,
) -> SelfQueryRetriever:
    """
    Create a SelfQueryRetriever instance from the given parameters.

    Args:
        model (BaseChatModel): LangChain chat model instance implementing the BaseChatModel interface.
        vector_store (VectorStore): Vector store instance to query.
        document_content_description (str): Short description of the document contents.
        metadata_field_info (list[AttributeInfo]): List of AttributeInfo describing metadata fields.
        verbose (bool): Whether to enable verbose logging (default False).

    Returns:
        SelfQueryRetriever.
    """
    retriever = SelfQueryRetriever.from_llm(
        model,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=verbose,
    )
    return retriever


def reorder_docs(
    docs: list[Document],
) -> list[Document]:
    """
    Reorder a list of documents using LongContextReorder transformer.

    Args:
        docs (list[Document]): List of LangChain Document objects.

    Returns:
        list[Document]: List of reordered documents.

    Raises:
        Exception: If an error occurs during the reordering process.
    """
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs


def create_retriever(
    vs: VectorStore,
    search_type: str,
    search_kwargs: dict,
) -> VectorStoreRetriever:
    """
    Create a retriever from a vectorstore.
    https://python.langchain.com/docs/how_to/vectorstore_retriever/

    Args:
        vs (VectorStore): Vectorstore instance used to create the retriever.
        search_type (str): Type of search to perform (e.g., 'similarity', 'similarity_score_threshold', 'mmr').
        search_kwargs (dict): Additional keyword arguments for the search (k, score_threshold, filter).

    Returns:
        VectorStoreRetriever: Instance of the retriever configured with the specified search type and parameters.
    """
    retriever = vs.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever


def create_BM25retriever_from_docs(
    docs: list[Document],
    k: int,
) -> BM25Retriever:
    """
    Create a BM25 retriever from a list of documents.

    Args:
        docs (list[Document]): Non-empty list of LangChain Document objects.
        k (int): Number of top documents to return per query.

    Returns:
        BM25Retriever: Configured BM25 retriever.

    Raises:
        ValueError: If docs is empty or k is not a positive integer.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    return bm25_retriever


def create_BM25retriever_from_text(
    docs: list[str],
    k: int,
) -> BM25Retriever:
    """
    Create a BM25 retriever from the provided texts.

    Args:
        docs (list[str]): List of strings.
        k (int): Number of top documents to retrieve.

    Returns:
        BM25Retriever: BM25 retriever instance configured with the provided texts and k value.

    Raises:
        ValueError: If the documents list is empty or if k is not a positive integer.
        Exception: For any other error that may occur during the BM25 retriever creation.
    """
    if not docs:
        raise ValueError("The documents list cannot be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    bm25_retriever = BM25Retriever.from_texts(docs)
    bm25_retriever.k = k
    return bm25_retriever


def create_hybrid_retriever(
    sparse_retriever,
    semantic_retriever,
    weights_sparse: float,
) -> EnsembleRetriever:
    """
    Create a hybrid retriever that combines a sparse retriever and a semantic retriever.

    Args:
        sparse_retriever: Instance of a sparse retriever.
        semantic_retriever: Instance of a semantic retriever.
        weights_sparse (float): The weight to assign to the sparse retriever,
                                which should be between 0 and 1.

    Returns:
        EnsembleRetriever: Ensemble retriever that combines the two retrievers.

    Raises:
        ValueError: If the weights_sparse is not between 0 and 1.
        Exception: For any other error that may occur during the hybrid retriever creation.
    """
    if not (0 <= weights_sparse <= 1):
        raise ValueError("weights_sparse must be between 0 and 1.")

    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, semantic_retriever],
        weights=[weights_sparse, 1 - weights_sparse],
    )
    return ensemble_retriever
