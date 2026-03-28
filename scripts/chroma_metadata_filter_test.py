"""
chroma_metadata_filter_test.py
===============================
Side-by-side comparison of two Chroma metadata-filtering strategies:

  Strategy A — Explicit filter
      Caller supplies a hard-coded filter dict via search_kwargs["filter"].
      Fast, deterministic, no LLM call at query time.

  Strategy B — SelfQueryRetriever
      An LLM translates the natural-language query into a structured filter
      + a semantic sub-query automatically.  More flexible, costs one LLM
      call per query.

Architecture
------------

  sample_docs (12 books with genre / year / author metadata)
         │
         ├── create_chroma_vectorstore(docs, embedding_fn)              ← build in-memory store
         ├── save_chroma_vectorstore(vs, persist_dir)                   ← write to disk
         └── load_chroma_vectorstore(persist_dir, embedding_fn)         ← reload from disk
                    │
                    ├── [A] create_retriever(vs, search_kwargs={"filter": {...}})
                    │         ^ hard-coded filter dict, caller is responsible
                    │
                    └── [B] create_self_query_retriever(model, vs, metadata_field_info)
                              ^ LLM parses natural-language query into filter + subquery

  Both retrievers operate on the same loaded Chroma collection so results
  are directly comparable and the full build → persist → load lifecycle
  is exercised end-to-end.

Invariants
----------
- OPENAI_API_KEY must be present in .env or the environment.
- The store is persisted to a temporary directory and cleaned up on exit.
- The _OpenAIEmbeddings shim bypasses langchain_openai.OpenAIEmbeddings to
  avoid version conflicts (same pattern as hybrid_rag.py).
- SelfQueryRetriever requires a Chroma backend — FAISS does not support
  structured metadata filtering in this version.

Debugging
---------
- Set LOG_LEVEL=DEBUG to see verbose Chroma and retriever logs.
- If SelfQuery returns 0 results, check the LLM-generated structured query
  by enabling verbose=True in create_self_query_retriever — it will print
  the filter + semantic sub-query to stdout.
- If the filter dict uses a field name that does not match metadata_field_info
  exactly, Chroma will silently return 0 results.
- Chroma filter syntax uses operators like "$eq", "$gt", "$in" — not plain
  dict equality at the top level when combining with other operators.
"""

import logging
import os
import tempfile

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI as _OpenAIClient

from kitai._langchain_compat import AttributeInfo
from kitai.index import (
    create_chroma_vectorstore,       # build ephemeral (in-memory) store
    load_chroma_vectorstore,         # reload a persisted store from disk
    save_chroma_vectorstore,         # persist an existing Chroma instance to disk
)
from kitai.retriever import create_retriever, create_self_query_retriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
K = 4  # number of docs to retrieve per query

# Metadata description consumed by SelfQueryRetriever to understand the corpus.
DOCUMENT_CONTENT_DESCRIPTION = "A short synopsis of a book."

METADATA_FIELD_INFO: list[AttributeInfo] = [
    AttributeInfo(
        name="genre",
        description="The genre of the book. One of: science_fiction, fantasy, history, thriller.",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the book was published.",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="The name of the book's author.",
        type="string",
    ),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI Embeddings shim (no langchain_openai.OpenAIEmbeddings dependency)
# ---------------------------------------------------------------------------

class _OpenAIEmbeddings(Embeddings):
    """Thin wrapper around the openai client satisfying langchain_core.embeddings.Embeddings."""

    def __init__(self, model: str, client: _OpenAIClient) -> None:
        self._model = model
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------

def build_sample_docs() -> list[Document]:
    """Return 12 synthetic book documents with genre / year / author metadata."""
    books = [
        # science_fiction
        ("A spacecraft crew discovers a signal from deep space that rewrites physics.",
         {"genre": "science_fiction", "year": 2021, "author": "Yara Chen", "id": "1"}),
        ("An AI achieves consciousness on a colony ship during a 200-year voyage.",
         {"genre": "science_fiction", "year": 2018, "author": "Marco Reyes", "id": "2"}),
        ("Earth faces extinction as a rogue planet approaches the solar system.",
         {"genre": "science_fiction", "year": 2015, "author": "Anika Patel", "id": "3"}),
        # fantasy
        ("A young mage uncovers a buried empire beneath the roots of an ancient forest.",
         {"genre": "fantasy", "year": 2022, "author": "Lena Voss", "id": "4"}),
        ("Two rival kingdoms must unite against a god awakening from a frozen sea.",
         {"genre": "fantasy", "year": 2019, "author": "Omar Fadel", "id": "5"}),
        ("A thief steals a cursed artifact that slowly erases her memories.",
         {"genre": "fantasy", "year": 2011, "author": "Soo-Jin Park", "id": "6"}),
        # history
        ("A detailed account of the fall of Constantinople and its lasting legacy.",
         {"genre": "history", "year": 2020, "author": "Pierre Blanc", "id": "7"}),
        ("The secret diplomatic channels that ended the Cuban Missile Crisis.",
         {"genre": "history", "year": 2009, "author": "Diana Torres", "id": "8"}),
        ("How the Silk Road shaped medieval economies across three continents.",
         {"genre": "history", "year": 2017, "author": "Ravi Sharma", "id": "9"}),
        # thriller
        ("A forensic accountant unravels a billion-dollar money-laundering scheme.",
         {"genre": "thriller", "year": 2023, "author": "Carla Bruni", "id": "10"}),
        ("A journalist disappears after exposing a surveillance program run by her own government.",
         {"genre": "thriller", "year": 2016, "author": "Jakub Novak", "id": "11"}),
        ("A retired spy is pulled back in when her former handler turns up dead.",
         {"genre": "thriller", "year": 2020, "author": "Maya Hill", "id": "12"}),
    ]
    return [Document(page_content=content, metadata=meta) for content, meta in books]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def run_explicit_filter(vs, query: str, filter_dict: dict) -> list[Document]:
    """Strategy A: retrieve with a hard-coded filter dict.

    Args:
        vs: Chroma vector store instance.
        query: Semantic search string.
        filter_dict: Chroma-compatible filter, e.g. {"genre": "fantasy"} or
            {"year": {"$gt": 2018}}.

    Returns:
        list[Document]: Top-K documents matching the filter.
    """
    retriever = create_retriever(
        vs=vs,
        search_type="similarity",
        search_kwargs={"k": K, "filter": filter_dict},
    )
    return retriever.invoke(query)


def run_self_query(vs, model, query: str, verbose: bool = False) -> list[Document]:
    """Strategy B: SelfQueryRetriever — LLM generates the filter from the query.

    Args:
        vs: Chroma vector store instance.
        model: LangChain BaseChatModel used to parse the query.
        query: Natural-language query; the LLM will extract any metadata
            constraints from it automatically.
        verbose: If True, prints the LLM-generated structured query.

    Returns:
        list[Document]: Top-K documents matching the auto-generated filter.
    """
    retriever = create_self_query_retriever(
        model=model,
        vector_store=vs,
        document_content_description=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        verbose=verbose,
    )
    return retriever.invoke(query)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(strategy: str, query: str, docs: list[Document]) -> None:
    """Print retrieved documents in a readable format."""
    print(f"\n{'─' * 70}")
    print(f"  {strategy}")
    print(f"  Query : {query!r}")
    print(f"{'─' * 70}")
    if not docs:
        print("  (no results)")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(
            f"  [{i}] [{meta.get('genre', '?')} | {meta.get('year', '?')} | {meta.get('author', '?')}]"
        )
        print(f"       {doc.page_content[:100]}…")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    {
        "label": "Genre filter — science fiction",
        "query": "space exploration and alien signals",
        # Strategy A: explicit filter
        "filter": {"genre": "science_fiction"},
        # Strategy B: self-query natural language (must contain the constraint)
        "self_query": "science fiction books about space exploration",
    },
    {
        "label": "Year filter — published after 2019",
        "query": "recent books",
        "filter": {"year": {"$gt": 2019}},
        "self_query": "books published after 2019",
    },
    {
        "label": "Combined filter — fantasy books after 2015",
        "query": "magic and kingdoms",
        "filter": {"$and": [{"genre": "fantasy"}, {"year": {"$gt": 2015}}]},
        "self_query": "fantasy books published after 2015",
    },
    {
        "label": "Author filter",
        "query": "conspiracy and surveillance",
        "filter": {"author": "Jakub Novak"},
        "self_query": "books by Jakub Novak",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )

    openai_client = _OpenAIClient(api_key=api_key)
    embedding_fn = _OpenAIEmbeddings(model=EMBEDDING_MODEL, client=openai_client)
    chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)

    docs = build_sample_docs()

    # ignore_cleanup_errors=True: on Windows, chromadb's hnswlib C++ destructor
    # holds data_level0.bin open until the process exits, so rmtree fails with
    # PermissionError (WinError 32).  The flag suppresses the error; temp files
    # are reclaimed by the OS when the process terminates.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as persist_dir:
        # Build the in-memory store, then persist it to disk.
        logger.info("Building Chroma store (%d docs) …", len(docs))
        vs_mem = create_chroma_vectorstore(
            docs=docs,
            embedding_fn=embedding_fn,
            collection_name="metadata_filter_test",
        )
        logger.info("Persisting Chroma store → %s …", persist_dir)
        save_chroma_vectorstore(vs_mem, persist_directory=persist_dir)

        # Reload from disk — this is the path that runs in production.
        logger.info("Loading Chroma store from disk …")
        vs = load_chroma_vectorstore(
            persist_directory=persist_dir,
            embedding_fn=embedding_fn,
            collection_name="metadata_filter_test",
        )
        logger.info("Chroma store ready.")

        # Run all test cases against the loaded store.
        for case in TEST_CASES:
            print(f"\n{'═' * 70}")
            print(f"  TEST: {case['label']}")
            print(f"{'═' * 70}")

            # Strategy A
            try:
                results_a = run_explicit_filter(vs, case["query"], case["filter"])
            except Exception as exc:
                logger.error("Strategy A failed: %s", exc)
                results_a = []
            print_results("Strategy A — Explicit filter", case["query"], results_a)

            # Strategy B
            try:
                results_b = run_self_query(vs, chat_model, case["self_query"])
            except Exception as exc:
                logger.error("Strategy B failed: %s", exc)
                results_b = []
            print_results("Strategy B — SelfQueryRetriever", case["self_query"], results_b)

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
