import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for tests — no API calls, dimension=4."""

    _DIM = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Each text gets a unique vector based on its hash so similarity
        # search produces non-trivial (but deterministic) rankings.
        return [
            [(hash(t) >> i & 0xFF) / 255.0 for i in range(self._DIM)]
            for t in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def mock_docs():
    return [
        Document(page_content=f"document {i}", metadata={"id": str(i)})
        for i in range(5)
    ]


@pytest.fixture
def fake_embeddings():
    return _FakeEmbeddings()
