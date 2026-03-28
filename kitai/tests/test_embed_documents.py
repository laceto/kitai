"""
kitai/tests/test_embed_documents.py
Tests for kitai.index.embed_documents.

Coverage:
  happy path          — returns float32 ndarray, shape (n_docs, dim)
  empty docs guard    — raises ValueError
  shape invariant     — one row per document
  dtype invariant     — always float32
  content invariant   — vectors match what embedding_fn produces directly

TDD cycle: tests written BEFORE implementation (Red → Green → Refactor).
"""

import numpy as np
import pytest
from langchain_core.documents import Document

from kitai.index import embed_documents


# ── happy path ────────────────────────────────────────────────────────────────

def test_embed_documents_returns_ndarray(mock_docs, fake_embeddings):
    result = embed_documents(mock_docs, fake_embeddings)
    assert isinstance(result, np.ndarray)


def test_embed_documents_shape(mock_docs, fake_embeddings):
    result = embed_documents(mock_docs, fake_embeddings)
    assert result.shape == (len(mock_docs), fake_embeddings._DIM)


def test_embed_documents_dtype_is_float32(mock_docs, fake_embeddings):
    result = embed_documents(mock_docs, fake_embeddings)
    assert result.dtype == np.float32


def test_embed_documents_single_doc(fake_embeddings):
    doc = Document(page_content="only one", metadata={"id": "1"})
    result = embed_documents([doc], fake_embeddings)
    assert result.shape == (1, fake_embeddings._DIM)


# ── content invariant ─────────────────────────────────────────────────────────

def test_embed_documents_values_match_embedding_fn(mock_docs, fake_embeddings):
    """Vectors must equal what embedding_fn.embed_documents produces directly."""
    result = embed_documents(mock_docs, fake_embeddings)
    expected = np.array(
        fake_embeddings.embed_documents([d.page_content for d in mock_docs]),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)


# ── guard: empty docs ─────────────────────────────────────────────────────────

def test_embed_documents_empty_docs_raises(fake_embeddings):
    with pytest.raises(ValueError, match="non-empty"):
        embed_documents([], fake_embeddings)
