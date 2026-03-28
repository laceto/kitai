import pytest
from pathlib import Path
from kitai.paths import check_and_create_folder


def test_create_new_folder(tmp_path):
    folder = tmp_path / "new_folder"
    result = check_and_create_folder(str(folder))
    assert result.exists() and result.is_dir()

def test_existing_folder(tmp_path):
    folder = tmp_path / "existing"
    folder.mkdir()
    result = check_and_create_folder(str(folder))
    assert result == folder

def test_invalid_input():
    with pytest.raises(ValueError):
        check_and_create_folder("")

def test_conflicting_file(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    with pytest.raises(OSError):
        check_and_create_folder(str(file_path))
