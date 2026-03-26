import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_docs():
    return [
        Document(page_content=f"document {i}", metadata={"id": str(i)})
        for i in range(5)
    ]
