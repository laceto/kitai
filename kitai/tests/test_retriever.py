"""
Tests for kitai.retriever.

Covers:
  - create_BM25retriever_from_docs  — happy path, empty docs, bad k
  - create_BM25retriever_from_text  — happy path, empty list, bad k
  - create_hybrid_retriever         — happy path, weight edges, bad weight
  - reorder_docs                    — happy path, preserves content, empty list
  - create_retriever                — happy path returning VectorStoreRetriever
"""

import pytest
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_BM25retriever_from_text,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs():
    return [Document(page_content=f"text {i}", metadata={"id": str(i)}) for i in range(5)]


@pytest.fixture
def bm25_retriever(sample_docs):
    return create_BM25retriever_from_docs(sample_docs, k=3)


@pytest.fixture
def chroma_retriever(sample_docs, fake_embeddings):
    """VectorStoreRetriever backed by an ephemeral Chroma store."""
    from kitai.index import create_chroma_vectorstore
    vs = create_chroma_vectorstore(sample_docs, fake_embeddings, collection_name="test")
    return create_retriever(vs, search_type="similarity", search_kwargs={"k": 2})


# ---------------------------------------------------------------------------
# create_BM25retriever_from_docs
# ---------------------------------------------------------------------------

def test_bm25_from_docs_returns_bm25_retriever(sample_docs):
    r = create_BM25retriever_from_docs(sample_docs, k=3)
    assert isinstance(r, BM25Retriever)


def test_bm25_from_docs_sets_k(sample_docs):
    r = create_BM25retriever_from_docs(sample_docs, k=2)
    assert r.k == 2


def test_bm25_from_docs_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        create_BM25retriever_from_docs([], k=3)


def test_bm25_from_docs_k_zero_raises(sample_docs):
    with pytest.raises(ValueError, match="positive"):
        create_BM25retriever_from_docs(sample_docs, k=0)


def test_bm25_from_docs_k_negative_raises(sample_docs):
    with pytest.raises(ValueError, match="positive"):
        create_BM25retriever_from_docs(sample_docs, k=-1)


# ---------------------------------------------------------------------------
# create_BM25retriever_from_text
# ---------------------------------------------------------------------------

def test_bm25_from_text_returns_bm25_retriever():
    r = create_BM25retriever_from_text(["hello", "world", "foo"], k=2)
    assert isinstance(r, BM25Retriever)


def test_bm25_from_text_sets_k():
    r = create_BM25retriever_from_text(["a", "b", "c"], k=1)
    assert r.k == 1


def test_bm25_from_text_empty_raises():
    with pytest.raises(ValueError):
        create_BM25retriever_from_text([], k=2)


def test_bm25_from_text_k_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        create_BM25retriever_from_text(["a", "b"], k=0)


# ---------------------------------------------------------------------------
# create_hybrid_retriever
# ---------------------------------------------------------------------------

def test_hybrid_retriever_returns_ensemble(bm25_retriever, chroma_retriever):
    r = create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=0.5)
    assert isinstance(r, EnsembleRetriever)


def test_hybrid_retriever_weights_sum_to_one(bm25_retriever, chroma_retriever):
    r = create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=0.4)
    assert abs(sum(r.weights) - 1.0) < 1e-9


def test_hybrid_retriever_sparse_weight_assigned(bm25_retriever, chroma_retriever):
    r = create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=0.3)
    assert r.weights[0] == pytest.approx(0.3)
    assert r.weights[1] == pytest.approx(0.7)


def test_hybrid_retriever_weight_zero_edge(bm25_retriever, chroma_retriever):
    r = create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=0.0)
    assert isinstance(r, EnsembleRetriever)


def test_hybrid_retriever_weight_one_edge(bm25_retriever, chroma_retriever):
    r = create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=1.0)
    assert isinstance(r, EnsembleRetriever)


def test_hybrid_retriever_weight_above_one_raises(bm25_retriever, chroma_retriever):
    with pytest.raises(ValueError, match="0 and 1"):
        create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=1.1)


def test_hybrid_retriever_weight_below_zero_raises(bm25_retriever, chroma_retriever):
    with pytest.raises(ValueError, match="0 and 1"):
        create_hybrid_retriever(bm25_retriever, chroma_retriever, weights_sparse=-0.1)


# ---------------------------------------------------------------------------
# reorder_docs
# ---------------------------------------------------------------------------

def test_reorder_docs_returns_list(sample_docs):
    result = reorder_docs(sample_docs)
    assert isinstance(result, list)


def test_reorder_docs_preserves_all_documents(sample_docs):
    result = reorder_docs(sample_docs)
    assert len(result) == len(sample_docs)
    original_contents = {d.page_content for d in sample_docs}
    reordered_contents = {d.page_content for d in result}
    assert original_contents == reordered_contents


def test_reorder_docs_returns_document_instances(sample_docs):
    result = reorder_docs(sample_docs)
    assert all(isinstance(d, Document) for d in result)


def test_reorder_docs_empty_list_returns_empty():
    assert reorder_docs([]) == []


# ---------------------------------------------------------------------------
# create_retriever
# ---------------------------------------------------------------------------

def test_create_retriever_returns_retriever(chroma_retriever):
    from langchain_core.vectorstores.base import VectorStoreRetriever
    assert isinstance(chroma_retriever, VectorStoreRetriever)


def test_create_retriever_respects_k(sample_docs, fake_embeddings):
    from kitai.index import create_chroma_vectorstore
    vs = create_chroma_vectorstore(sample_docs, fake_embeddings, collection_name="test-k")
    retriever = create_retriever(vs, search_type="similarity", search_kwargs={"k": 2})
    results = retriever.invoke("text 0")
    assert len(results) <= 2
