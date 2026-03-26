"""
Deprecation-warning tests for the three disk-based batch functions in kitai.index.

These functions are superseded by kitai.batch but kept for backward compatibility.
Each must emit a DeprecationWarning at call time while still working correctly.
"""
import warnings
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


def _make_doc(id_: int) -> Document:
    return Document(page_content=f"text {id_}", metadata={"id": str(id_)})


# ── create_batch_files_embeddings ─────────────────────────────────────────────

def test_create_batch_files_embeddings_emits_deprecation(tmp_path):
    from kitai.index import create_batch_files_embeddings

    docs = [_make_doc(0), _make_doc(1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_batch_files_embeddings(docs, output_dir=str(tmp_path))

    categories = [w.category for w in caught]
    assert DeprecationWarning in categories, (
        f"Expected DeprecationWarning; got: {categories}"
    )


def test_create_batch_files_embeddings_still_works(tmp_path):
    """Deprecation warning must not break existing behaviour."""
    from kitai.index import create_batch_files_embeddings

    docs = [_make_doc(0), _make_doc(1), _make_doc(2)]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        create_batch_files_embeddings(
            docs, batch_size=2, output_dir=str(tmp_path), batch_file_name="t"
        )

    files = sorted(tmp_path.glob("t_part*.jsonl"))
    assert len(files) == 2
    assert all(f.read_text(encoding="utf-8").strip() != "" for f in files)


def test_create_batch_files_embeddings_deprecation_message(tmp_path):
    """Warning message must mention the replacement API."""
    from kitai.index import create_batch_files_embeddings

    docs = [_make_doc(0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_batch_files_embeddings(docs, output_dir=str(tmp_path))

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "No DeprecationWarning emitted"
    assert "kitai.batch" in str(dep_warnings[0].message).lower() or \
           "build_embedding_tasks" in str(dep_warnings[0].message)


# ── create_embeddings_batches ─────────────────────────────────────────────────

def test_create_embeddings_batches_emits_deprecation(tmp_path):
    from kitai.index import create_embeddings_batches

    mock_client = MagicMock()
    mock_client.files.create.return_value = MagicMock(id="file_1")
    mock_client.batches.create.return_value = MagicMock(id="batch_1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Pass an empty real directory so the function doesn't raise ValueError
        create_embeddings_batches(mock_client, str(tmp_path))

    categories = [w.category for w in caught]
    assert DeprecationWarning in categories, (
        f"Expected DeprecationWarning; got: {categories}"
    )


# ── retrieve_embeddings_batches ───────────────────────────────────────────────

def test_retrieve_embeddings_batches_emits_deprecation():
    from kitai.index import retrieve_embeddings_batches

    mock_client = MagicMock()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Empty job_ids list — no network call, but warning still fires
        retrieve_embeddings_batches(mock_client, [])

    categories = [w.category for w in caught]
    assert DeprecationWarning in categories, (
        f"Expected DeprecationWarning; got: {categories}"
    )
