"""
kitai/tests/test_chroma.py
Tests for Chroma vector store factory functions in kitai.index.

Coverage:
  create_chroma_vectorstore                — happy path, empty-docs guard, empty-collection-name guard
  save_chroma_vectorstore                  — happy path (writes to disk), empty-docs guard, empty-dir guard
  load_chroma_vectorstore                  — happy path (reads from disk), missing-dir guard
  create_chroma_vectorstore_from_embeddings — happy path, length mismatch, missing id, duplicate id,
                                             float32 ndarray acceptance, persist_directory round-trip

TDD cycle: all tests written BEFORE implementation. Run pytest to confirm Red.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings


# ── helpers ───────────────────────────────────────────────────────────────────

EMBEDDING_SIZE = 8


def _fake_embedding_fn():
    return FakeEmbeddings(size=EMBEDDING_SIZE)


def _make_docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"doc {i}", metadata={"id": str(i), "title": f"Title {i}"})
        for i in range(n)
    ]


# ── create_chroma_vectorstore ─────────────────────────────────────────────────


def test_create_chroma_vectorstore_returns_chroma():
    from langchain_community.vectorstores import Chroma
    from kitai.index import create_chroma_vectorstore

    vs = create_chroma_vectorstore(_make_docs(), _fake_embedding_fn())
    assert isinstance(vs, Chroma)


def test_create_chroma_vectorstore_similarity_search():
    from kitai.index import create_chroma_vectorstore

    docs = _make_docs(4)
    vs = create_chroma_vectorstore(docs, _fake_embedding_fn())
    results = vs.similarity_search("doc 0", k=2)
    assert len(results) == 2


def test_create_chroma_vectorstore_empty_docs_raises():
    from kitai.index import create_chroma_vectorstore

    with pytest.raises(ValueError, match="docs"):
        create_chroma_vectorstore([], _fake_embedding_fn())


def test_create_chroma_vectorstore_empty_collection_name_raises():
    from kitai.index import create_chroma_vectorstore

    with pytest.raises(ValueError, match="collection_name"):
        create_chroma_vectorstore(_make_docs(), _fake_embedding_fn(), collection_name="")


# ── save_chroma_vectorstore ───────────────────────────────────────────────────


def test_save_chroma_vectorstore_writes_to_disk(tmp_path):
    from kitai.index import save_chroma_vectorstore

    persist_dir = str(tmp_path / "chroma_db")
    save_chroma_vectorstore(_make_docs(), _fake_embedding_fn(), persist_directory=persist_dir)

    # Chroma creates files in the persist directory
    assert Path(persist_dir).exists()
    assert any(Path(persist_dir).iterdir())


def test_save_chroma_vectorstore_returns_chroma(tmp_path):
    from langchain_community.vectorstores import Chroma
    from kitai.index import save_chroma_vectorstore

    persist_dir = str(tmp_path / "chroma_db")
    vs = save_chroma_vectorstore(_make_docs(), _fake_embedding_fn(), persist_directory=persist_dir)
    assert isinstance(vs, Chroma)


def test_save_chroma_vectorstore_empty_docs_raises(tmp_path):
    from kitai.index import save_chroma_vectorstore

    with pytest.raises(ValueError, match="docs"):
        save_chroma_vectorstore([], _fake_embedding_fn(), persist_directory=str(tmp_path))


def test_save_chroma_vectorstore_empty_dir_raises():
    from kitai.index import save_chroma_vectorstore

    with pytest.raises(ValueError, match="persist_directory"):
        save_chroma_vectorstore(_make_docs(), _fake_embedding_fn(), persist_directory="")


# ── load_chroma_vectorstore ───────────────────────────────────────────────────


def test_load_chroma_vectorstore_round_trip(tmp_path):
    from kitai.index import save_chroma_vectorstore, load_chroma_vectorstore

    persist_dir = str(tmp_path / "chroma_db")
    docs = _make_docs(3)
    save_chroma_vectorstore(docs, _fake_embedding_fn(), persist_directory=persist_dir)

    loaded = load_chroma_vectorstore(persist_dir, _fake_embedding_fn())
    results = loaded.similarity_search("doc 1", k=1)
    assert len(results) == 1


def test_load_chroma_vectorstore_empty_dir_raises():
    from kitai.index import load_chroma_vectorstore

    with pytest.raises(ValueError, match="persist_directory"):
        load_chroma_vectorstore("", _fake_embedding_fn())


def test_load_chroma_vectorstore_missing_dir_raises(tmp_path):
    from kitai.index import load_chroma_vectorstore

    with pytest.raises(FileNotFoundError):
        load_chroma_vectorstore(str(tmp_path / "nonexistent"), _fake_embedding_fn())


# ── create_chroma_vectorstore_from_embeddings ─────────────────────────────────

import numpy as np


def _make_embeddings(n: int = 3) -> np.ndarray:
    """Deterministic float32 ndarray — same shape convention as kitai.batch output."""
    rng = np.random.default_rng(seed=7)
    return rng.random((n, EMBEDDING_SIZE)).astype(np.float32)


def test_from_embeddings_returns_chroma():
    from langchain_community.vectorstores import Chroma
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = _make_docs(3)
    embs = _make_embeddings(3)
    vs = create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())
    assert isinstance(vs, Chroma)


def test_from_embeddings_similarity_search():
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = _make_docs(4)
    embs = _make_embeddings(4)
    vs = create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())
    results = vs.similarity_search("doc 0", k=2)
    assert len(results) == 2


def test_from_embeddings_accepts_float32_ndarray():
    """float32 must be accepted without TypeError — internal cast handles it."""
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = _make_docs(3)
    embs = _make_embeddings(3)
    assert embs.dtype == np.float32  # confirm input is float32
    vs = create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())
    assert vs.similarity_search("doc 0", k=1)  # no exception means cast worked


def test_from_embeddings_length_mismatch_raises():
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = _make_docs(3)
    embs = _make_embeddings(5)  # wrong length
    with pytest.raises(ValueError, match="same length"):
        create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())


def test_from_embeddings_empty_docs_raises():
    from kitai.index import create_chroma_vectorstore_from_embeddings

    with pytest.raises(ValueError, match="docs"):
        create_chroma_vectorstore_from_embeddings(
            [], _make_embeddings(0), _fake_embedding_fn()
        )


def test_from_embeddings_missing_id_raises():
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = [Document(page_content="no id here", metadata={"title": "x"})]
    embs = _make_embeddings(1)
    with pytest.raises((ValueError, KeyError)):
        create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())


def test_from_embeddings_duplicate_id_raises():
    from kitai.index import create_chroma_vectorstore_from_embeddings

    docs = [
        Document(page_content="a", metadata={"id": "1"}),
        Document(page_content="b", metadata={"id": "1"}),  # duplicate
    ]
    embs = _make_embeddings(2)
    with pytest.raises(ValueError, match="duplicate"):
        create_chroma_vectorstore_from_embeddings(docs, embs, _fake_embedding_fn())


def test_from_embeddings_persist_round_trip(tmp_path):
    """Vectors saved to disk are retrievable in a new Chroma instance."""
    from kitai.index import create_chroma_vectorstore_from_embeddings, load_chroma_vectorstore

    persist_dir = str(tmp_path / "chroma_emb")
    docs = _make_docs(3)
    embs = _make_embeddings(3)
    collection = "emb_test"

    create_chroma_vectorstore_from_embeddings(
        docs, embs, _fake_embedding_fn(),
        collection_name=collection,
        persist_directory=persist_dir,
    )

    loaded = load_chroma_vectorstore(persist_dir, _fake_embedding_fn(), collection_name=collection)
    results = loaded.similarity_search("doc 1", k=1)
    assert len(results) == 1
