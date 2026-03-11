"""
hybrid_rag.py
=============
Load an existing FAISS vector store, build a hybrid retriever (BM25 + semantic),
run a query, and return an LLM-generated answer.

Architecture
------------
                     ┌─ BM25Retriever          (keyword)
  FAISS .faiss/.pkl  │
         │           └─ VectorStoreRetriever   (semantic)
         │                     │
         └─────────────────────┤
                               ▼
                       EnsembleRetriever  (RRF merge, weights_sparse=0.5)
                               │
                          reorder_docs()  (LongContextReorder)
                               │
                         context string
                               │
                           ChatOpenAI
                               │
                            answer

Invariants
----------
- OPENAI_API_KEY must be present in .env (or the environment).
- The vectorstore folder must contain both the .faiss and .pkl files
  named after FAISS_INDEX_NAME.
- Docs are extracted from the loaded FAISS docstore; the same corpus
  is used for both the vector retriever and BM25 — they are never out of sync.

Debugging
---------
- Set LOG_LEVEL=DEBUG for verbose retrieval logs (BM25 scores, RRF ranks).
- If FAISS raises a deserialization error, confirm allow_dangerous_deserialization=True
  is acceptable — it is safe here because we control the .pkl file.
- If the answer is poor, try lowering weights_sparse toward 0.0 (more semantic)
  or raising it toward 1.0 (more keyword-driven).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI as _OpenAIClient

from kitai.retriever import (
    create_BM25retriever_from_docs,
    create_hybrid_retriever,
    create_retriever,
    reorder_docs,
)

# ---------------------------------------------------------------------------
# Embeddings shim — avoids langchain_openai (incompatible with langchain-core 0.3.x)
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
# Configuration — change these constants to adapt the script
# ---------------------------------------------------------------------------

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore" / "rss_feeds"
FAISS_INDEX_NAME = "faiss_index_rss_feeds"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

K_SEMANTIC = 6   # docs to retrieve from FAISS
K_BM25 = 6       # docs to retrieve from BM25
WEIGHTS_SPARSE = 0.5  # 0.0 = pure semantic, 1.0 = pure BM25

QUERY = "What are the latest news about Iran - USA war?"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_vectorstore(
    folder: Path,
    index_name: str,
    embeddings_model: _OpenAIEmbeddings,
) -> FAISS:
    """Load a FAISS index from disk.

    Raises:
        FileNotFoundError: If .faiss or .pkl files are missing.
    """
    faiss_file = folder / f"{index_name}.faiss"
    pkl_file = folder / f"{index_name}.pkl"

    for f in (faiss_file, pkl_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Expected vectorstore file not found: {f}\n"
                f"Run the embedding pipeline first to create the index."
            )

    logger.info("Loading FAISS index from %s …", folder)
    vs = FAISS.load_local(
        folder_path=str(folder),
        embeddings=embeddings_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # safe: we own this .pkl
    )
    logger.info("Loaded %d vectors.", vs.index.ntotal)
    return vs


def extract_docs(vs: FAISS) -> list[Document]:
    """Pull all Documents stored in the FAISS in-memory docstore.

    These are used to build the BM25 index over the same corpus, so
    both retrievers always see identical documents.
    """
    return list(vs.docstore._dict.values())


def build_hybrid_retriever(vs: FAISS, docs: list[Document]):
    """Wire together BM25 + semantic retriever via EnsembleRetriever."""
    bm25 = create_BM25retriever_from_docs(docs=docs, k=K_BM25)
    vector = create_retriever(
        vs=vs,
        search_type="similarity",
        search_kwargs={"k": K_SEMANTIC},
    )
    hybrid = create_hybrid_retriever(
        sparse_retriever=bm25,
        semantic_retriever=vector,
        weights_sparse=WEIGHTS_SPARSE,
    )
    logger.info(
        "Hybrid retriever ready (BM25 k=%d, semantic k=%d, weights=%s).",
        K_BM25,
        K_SEMANTIC,
        hybrid.weights,
    )
    return hybrid


def answer_query(query: str, context_docs: list[Document], client: _OpenAIClient) -> str:
    """Send the reordered context + query to the LLM and return the answer."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = (
        f"Use the following news excerpts to answer the question.\n"
        f"If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )

    # 1. Load vectorstore
    openai_client = _OpenAIClient(api_key=api_key)
    embeddings_model = _OpenAIEmbeddings(model=EMBEDDING_MODEL, client=openai_client)
    vs = load_vectorstore(VECTORSTORE_DIR, FAISS_INDEX_NAME, embeddings_model)

    # 2. Extract corpus for BM25 (same docs as FAISS — always in sync)
    docs = extract_docs(vs)
    logger.info("Corpus size: %d documents.", len(docs))

    # 3. Build hybrid retriever
    hybrid = build_hybrid_retriever(vs, docs)

    # 4. Retrieve + reorder
    logger.info("Querying: %r", QUERY)
    retrieved = hybrid.invoke(QUERY)
    ordered = reorder_docs(retrieved)
    logger.info("Retrieved %d documents; reordered for LLM context.", len(ordered))

    # 5. Generate answer
    answer = answer_query(QUERY, ordered, openai_client)

    # 6. Print results
    print("\n" + "=" * 70)
    print(f"QUERY : {QUERY}")
    print("=" * 70)
    print("\nSOURCE DOCUMENTS (LongContextReorder applied):")
    for i, doc in enumerate(ordered, 1):
        snippet = doc.page_content[:120].replace("\n", " ")
        meta = {k: v for k, v in doc.metadata.items() if k != "id"}
        print(f"  [{i}] {snippet}…")
        if meta:
            print(f"       metadata: {meta}")
    print("\nANSWER:")
    print(answer)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
