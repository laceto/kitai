import logging
import pytest
import pandas as pd

import kitai.export as export_mod
from kitai.export import df_to_csv, df_to_excel


@pytest.fixture
def simple_df():
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


# ── logger presence ───────────────────────────────────────────────────────────

def test_export_has_module_logger():
    assert hasattr(export_mod, "logger"), "export.py is missing a module-level 'logger'"
    assert export_mod.logger.name == "kitai.export"


# ── df_to_csv ─────────────────────────────────────────────────────────────────

def test_df_to_csv_no_print(tmp_path, capsys, simple_df):
    path = str(tmp_path / "out.csv")
    df_to_csv(simple_df, path)
    captured = capsys.readouterr()
    assert captured.out == "", f"df_to_csv printed to stdout: {captured.out!r}"


def test_df_to_csv_logs_on_success(tmp_path, caplog, simple_df):
    path = str(tmp_path / "out.csv")
    with caplog.at_level(logging.INFO, logger="kitai.export"):
        df_to_csv(simple_df, path)
    assert len(caplog.records) >= 1
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_df_to_csv_writes_file(tmp_path, simple_df):
    path = str(tmp_path / "out.csv")
    df_to_csv(simple_df, path)
    import os
    assert os.path.exists(path)
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["a", "b"]


# ── df_to_excel ───────────────────────────────────────────────────────────────

def test_df_to_excel_no_print(tmp_path, capsys, simple_df):
    path = str(tmp_path / "out.xlsx")
    df_to_excel(simple_df, path)
    captured = capsys.readouterr()
    assert captured.out == "", f"df_to_excel printed to stdout: {captured.out!r}"


def test_df_to_excel_logs_on_success(tmp_path, caplog, simple_df):
    path = str(tmp_path / "out.xlsx")
    with caplog.at_level(logging.INFO, logger="kitai.export"):
        df_to_excel(simple_df, path)
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_df_to_excel_writes_file(tmp_path, simple_df):
    path = str(tmp_path / "out.xlsx")
    df_to_excel(simple_df, path)
    import os
    assert os.path.exists(path)
    loaded = pd.read_excel(path)
    assert list(loaded.columns) == ["a", "b"]
