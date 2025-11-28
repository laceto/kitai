import pytest
from pathlib import Path
from mymodule import check_and_create_folder

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

def test_empty_docs(tmp_path):
    with pytest.raises(ValueError):
        create_batch_files([], batch_size=10, output_dir=tmp_path)

def test_batch_creation(tmp_path, mock_docs):
    create_batch_files(mock_docs, batch_size=2, output_dir=tmp_path, batch_file_name="test_batch")
    files = list(tmp_path.glob("test_batch_part*.jsonl"))
    assert len(files) == (len(mock_docs) + 1) // 2
    for f in files:
        assert f.read_text().strip() != ""
