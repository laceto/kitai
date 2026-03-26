import logging
import pytest
from langchain_core.documents import Document

import kitai.transform as transform_mod
from kitai.transform import (
    list_to_docs,
    flatten_list_of_lists,
    add_num_id_to_metadata,
)


# ── Commit 1: module-level logger ────────────────────────────────────────────

def test_transform_has_module_logger():
    """transform.py must expose a module-level logger named 'kitai.transform'."""
    assert hasattr(transform_mod, "logger"), "transform.py is missing a module-level 'logger'"
    assert transform_mod.logger.name == "kitai.transform"


# ── Commit 2: add_num_id_to_metadata ─────────────────────────────────────────

def test_add_num_id_assigns_to_all_docs():
    """All documents must receive id_new, not just the first one."""
    docs = [
        Document(page_content="a"),
        Document(page_content="b"),
        Document(page_content="c"),
    ]
    result = add_num_id_to_metadata(docs)
    assert len(result) == 3
    for i, doc in enumerate(result):
        assert "id_new" in doc.metadata, f"doc {i} missing id_new"
        assert doc.metadata["id_new"] == i, (
            f"doc {i} has id_new={doc.metadata['id_new']}, expected {i}"
        )


def test_add_num_id_single_doc():
    docs = [Document(page_content="only")]
    result = add_num_id_to_metadata(docs)
    assert result[0].metadata["id_new"] == 0


def test_add_num_id_preserves_existing_metadata():
    docs = [Document(page_content="x", metadata={"topic": "risk"})]
    result = add_num_id_to_metadata(docs)
    assert result[0].metadata["topic"] == "risk"
    assert result[0].metadata["id_new"] == 0


# ── Commit 3: list_to_docs — remove bare except + print ──────────────────────

def test_list_to_docs_happy_path():
    docs = list_to_docs(["hello", "world"])
    assert len(docs) == 2
    assert docs[0].page_content == "hello"
    assert docs[1].page_content == "world"


def test_list_to_docs_empty_raises():
    """Empty input must raise ValueError, not swallow it."""
    with pytest.raises(ValueError, match="empty"):
        list_to_docs([])


def test_list_to_docs_non_string_raises():
    """Non-string elements must raise ValueError, not swallow it."""
    with pytest.raises(ValueError):
        list_to_docs(["valid", 42])


def test_list_to_docs_no_print_on_error(capsys):
    """No text must be printed to stdout on error."""
    with pytest.raises(ValueError):
        list_to_docs([])
    captured = capsys.readouterr()
    assert captured.out == "", f"list_to_docs printed to stdout: {captured.out!r}"


# ── Commit 3: flatten_list_of_lists — remove bare except + print ─────────────

def test_flatten_happy_path():
    result = flatten_list_of_lists([[1, 2], [3, 4]])
    assert result == [1, 2, 3, 4]


def test_flatten_empty_list():
    assert flatten_list_of_lists([]) == []


def test_flatten_raises_on_non_list_of_lists():
    """Flat list input must raise TypeError, not print and return None."""
    with pytest.raises(TypeError, match="list of lists"):
        flatten_list_of_lists([1, 2, 3])


def test_flatten_no_print_on_error(capsys):
    with pytest.raises(TypeError):
        flatten_list_of_lists([1, 2, 3])
    captured = capsys.readouterr()
    assert captured.out == "", f"flatten_list_of_lists printed to stdout: {captured.out!r}"
