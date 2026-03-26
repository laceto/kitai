"""
kitai/tests/test_batch.py
Tests for chat completion batch helpers in kitai.batch.

Coverage:
  build_chat_tasks  — happy path, empty guard, missing-id KeyError
  parse_chat_results — happy path, item-level error skip, bad shape skip, empty guard
"""

import pytest
from kitai.batch import build_chat_tasks, parse_chat_results

# ── build_chat_tasks ──────────────────────────────────────────────────────────


def _make_item(id_: str, content: str) -> dict:
    return {"id": id_, "content": content}


def test_build_chat_tasks_happy_path():
    items = [_make_item("1", "Hello"), _make_item("2", "World")]
    tasks = build_chat_tasks(items, system_prompt="You are helpful.")

    assert len(tasks) == 2

    task = tasks[0]
    assert task["custom_id"] == "custom_id_1"
    assert task["method"] == "POST"
    assert task["url"] == "/v1/chat/completions"

    body = task["body"]
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    assert "model" in body


def test_build_chat_tasks_default_model():
    tasks = build_chat_tasks([_make_item("x", "hi")], system_prompt="sys")
    assert tasks[0]["body"]["model"] == "gpt-4o-mini"


def test_build_chat_tasks_custom_model():
    tasks = build_chat_tasks(
        [_make_item("x", "hi")], system_prompt="sys", model="gpt-4o"
    )
    assert tasks[0]["body"]["model"] == "gpt-4o"


def test_build_chat_tasks_empty_raises():
    with pytest.raises(ValueError, match="items"):
        build_chat_tasks([], system_prompt="sys")


def test_build_chat_tasks_missing_id_raises():
    with pytest.raises(KeyError):
        build_chat_tasks([{"content": "no id here"}], system_prompt="sys")


def test_build_chat_tasks_missing_content_raises():
    with pytest.raises(KeyError):
        build_chat_tasks([{"id": "1"}], system_prompt="sys")


# ── parse_chat_results ────────────────────────────────────────────────────────


def _make_result(custom_id: str, content: str) -> dict:
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": content}}]
            },
        },
    }


def _make_error_result(custom_id: str, error_msg: str) -> dict:
    return {
        "custom_id": custom_id,
        "error": {"message": error_msg},
        "response": None,
    }


def _make_bad_shape_result(custom_id: str) -> dict:
    """Result with unexpected response structure (no choices)."""
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {"status_code": 200, "body": {}},
    }


def test_parse_chat_results_happy_path():
    results = [
        _make_result("custom_id_1", "answer one"),
        _make_result("custom_id_2", "answer two"),
    ]
    parsed = parse_chat_results(results, extractor_fn=str.upper)

    assert len(parsed) == 2
    assert parsed[0] == ("custom_id_1", "ANSWER ONE")
    assert parsed[1] == ("custom_id_2", "ANSWER TWO")


def test_parse_chat_results_identity_extractor():
    results = [_make_result("custom_id_3", "raw text")]
    parsed = parse_chat_results(results, extractor_fn=lambda x: x)
    assert parsed[0] == ("custom_id_3", "raw text")


def test_parse_chat_results_skips_item_error(caplog):
    results = [
        _make_error_result("custom_id_bad", "rate limit"),
        _make_result("custom_id_good", "ok"),
    ]
    import logging
    with caplog.at_level(logging.ERROR, logger="kitai.batch"):
        parsed = parse_chat_results(results, extractor_fn=lambda x: x)

    assert len(parsed) == 1
    assert parsed[0][0] == "custom_id_good"
    assert "custom_id_bad" in caplog.text


def test_parse_chat_results_skips_bad_shape(caplog):
    results = [
        _make_bad_shape_result("custom_id_broken"),
        _make_result("custom_id_ok", "fine"),
    ]
    import logging
    with caplog.at_level(logging.ERROR, logger="kitai.batch"):
        parsed = parse_chat_results(results, extractor_fn=lambda x: x)

    assert len(parsed) == 1
    assert parsed[0][0] == "custom_id_ok"
    assert "custom_id_broken" in caplog.text


def test_parse_chat_results_extractor_exception_skipped(caplog):
    """Extractor errors are treated as per-item failures — logged and skipped."""
    def boom(text: str) -> str:
        raise ValueError("parse failed")

    results = [
        _make_result("custom_id_x", "some text"),
        _make_result("custom_id_y", "other text"),
    ]
    import logging
    with caplog.at_level(logging.ERROR, logger="kitai.batch"):
        parsed = parse_chat_results(results, extractor_fn=boom)

    assert parsed == []
    assert "custom_id_x" in caplog.text


def test_parse_chat_results_empty_raises():
    with pytest.raises(ValueError, match="results"):
        parse_chat_results([], extractor_fn=lambda x: x)
