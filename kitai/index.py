from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging
from pathlib import Path
import ast
import numpy as np
import pandas as pd

from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

logger = logging.getLogger(__name__)

def create_faiss_vectorstore_from_embeddings(
    docs: list[Document],
    embeddings: np.ndarray,
    query_encoder,
) -> FAISS:
    """
    Build a FAISS vector store from pre-computed embeddings and documents.

    Invariants:
        - ``len(docs)`` must equal ``embeddings.shape[0]``.  A mismatch means
          the i-th embedding would be paired with the wrong document.
        - Every document must have ``doc.metadata["id"]`` set; it is used as the
          key in the in-memory docstore and in ``index_to_docstore_id``.

    Args:
        docs (list[Document]): Documents to store.  Must be the same length as
            ``embeddings`` and each must carry a ``metadata["id"]`` field.
        embeddings (np.ndarray): 2-D float array of shape (n_docs, embedding_dim)
            with pre-computed embeddings in the same order as ``docs``.
        query_encoder: A LangChain embeddings object stored as FAISS's
            ``embedding_function``.  Not called during construction (embeddings
            are pre-computed); only used at query time to re-encode query strings.

    Returns:
        FAISS: Populated FAISS vector store ready for similarity search.

    Raises:
        ValueError: If ``len(docs) != embeddings.shape[0]``.
    """
    if len(docs) != embeddings.shape[0]:
        raise ValueError(
            f"docs and embeddings must have the same length. "
            f"Got {len(docs)} docs and {embeddings.shape[0]} embedding rows."
        )

    embedding_dim = get_embedding_dim(embeddings)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    index_to_docstore_id = {i: doc.metadata["id"] for i, doc in enumerate(docs)}
    docstore = InMemoryDocstore({doc.metadata["id"]: doc for doc in docs})

    return FAISS(
        embedding_function=query_encoder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


# Backward-compat alias — prefer create_faiss_vectorstore_from_embeddings in new code.
create_vectorstore = create_faiss_vectorstore_from_embeddings


def get_embedding_dim(embeddings: np.ndarray) -> int:
    """
    Extract embedding dimension from a numpy array.

    Args:
        embeddings (np.ndarray): A 2D numpy array of shape (n_samples, n_features).

    Returns:
        int: The embedding dimension (number of features).
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {embeddings.ndim}D array instead.")
    
    return embeddings.shape[1]

def load_embeddings_from_csv(
    path_to_csv: str = './book_embeddings.csv',
    embedding_column: str = 'embedding',
) -> np.ndarray:
    """
    Load embeddings from a CSV file.

    Args:
        path_to_csv: Path to CSV file containing embeddings.
        embedding_column: Column name with embedding data.

    Returns:
        np.ndarray: Stacked embeddings array of shape (n_rows, embedding_dim).

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        KeyError: If embedding_column not in CSV.
        ValueError: If embedding format is invalid.
    """
    try:
        # Load CSV with explicit error handling
        df_embeddings = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path_to_csv}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")
    
    # Verify column exists
    if embedding_column not in df_embeddings.columns:
        raise KeyError(
            f"Column '{embedding_column}' not found. "
            f"Available columns: {list(df_embeddings.columns)}"
        )
    
    # Parse embeddings with validation
    def parse_embedding(x: str) -> np.ndarray:
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Invalid embedding format. Expected list or array string. "
                f"Got: {x[:50]}... Error: {str(e)}"
            )
    
    df_embeddings["embedding_array"] = df_embeddings[embedding_column].apply(
        parse_embedding
    )
    
    # Stack embeddings
    embeddings = df_embeddings['embedding_array']
    return np.stack(embeddings)


def embed_documents(
    docs: list[Document],
    embedding_fn: Embeddings,
) -> np.ndarray:
    """
    Encode documents synchronously and return a float32 ndarray.

    This is the synchronous counterpart to the batch-API workflow in
    ``kitai.batch``.  Use it for small corpora where results are needed
    immediately; for large corpora prefer ``build_embedding_tasks`` +
    ``submit_batch_job`` (50 % cheaper, async).

    The returned array feeds directly into the vectorstore constructors::

        embeddings = embed_documents(docs, embedding_fn)
        vs = create_faiss_vectorstore_from_embeddings(docs, embeddings, embedding_fn)
        vs = create_chroma_vectorstore_from_embeddings(docs, embeddings, embedding_fn)

    Args:
        docs: Documents whose ``page_content`` will be embedded.
        embedding_fn: Any LangChain ``Embeddings`` instance
            (e.g. ``OpenAIEmbeddings``, ``FakeEmbeddings``).
            The caller is responsible for initialising the model.

    Returns:
        ``np.ndarray`` of shape ``(len(docs), embedding_dim)``, dtype ``float32``.

    Raises:
        ValueError: If ``docs`` is empty.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")
    texts = [doc.page_content for doc in docs]
    vectors = embedding_fn.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def create_chroma_vectorstore(
    docs: list[Document],
    embedding_fn,
    collection_name: str = "kitai",
) -> Chroma:
    """
    Build an ephemeral (in-memory) Chroma vector store from a list of documents.

    Use this when you need a vector store that supports ``SelfQueryRetriever``
    structured metadata filtering — FAISS does not support it.

    Invariants:
        - ``docs`` must be non-empty.
        - ``collection_name`` must be a non-empty string.

    Args:
        docs (list[Document]): Documents to index.  Each should carry a
            ``metadata["id"]`` field if you intend to filter by ID later.
        embedding_fn: A LangChain ``Embeddings`` instance (e.g. ``OpenAIEmbeddings()``)
            used to encode both documents and queries.
        collection_name (str): Chroma collection name.  Defaults to ``"kitai"``.

    Returns:
        Chroma: Populated in-memory Chroma vector store.

    Raises:
        ValueError: If ``docs`` is empty or ``collection_name`` is blank.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")
    if not collection_name:
        raise ValueError("collection_name must be a non-empty string.")

    return Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,
        collection_name=collection_name,
    )


def create_chroma_vectorstore_from_embeddings(
    docs: list[Document],
    embeddings: np.ndarray,
    query_encoder,
    collection_name: str = "kitai",
    persist_directory: str | None = None,
) -> Chroma:
    """
    Build a Chroma vector store from pre-computed embeddings and documents.

    Mirrors the ``create_faiss_vectorstore_from_embeddings`` workflow: callers
    supply embeddings they have already computed (e.g. via ``kitai.batch``) and
    get a Chroma store back — no second call to the embedding API.

    In chromadb 1.5+, ``Chroma.from_embeddings`` no longer exists.  This
    function creates an empty Chroma collection and injects the pre-computed
    vectors via ``_collection.upsert`` — which accepts ``np.ndarray`` directly
    and does **not** call ``embedding_function`` during construction.
    ``query_encoder`` is only stored for query-time re-encoding of search strings.

    Invariants:
        - ``docs`` must be non-empty.
        - ``len(docs)`` must equal ``embeddings.shape[0]``.
        - Every ``doc.metadata["id"]`` must be set and unique across the list.
        - ``collection_name`` must be non-empty.

    Args:
        docs (list[Document]): Documents to store, in the same order as
            ``embeddings``.  Each must carry ``metadata["id"]``.
        embeddings (np.ndarray): 2-D array of shape ``(n_docs, embedding_dim)``.
            ``float32`` and ``float64`` are both accepted; internally cast to
            ``float64`` before passing to Chroma to avoid numpy scalar type
            errors.
        query_encoder: A LangChain ``Embeddings`` instance stored for query-time
            use only (not called during construction).
        collection_name (str): Chroma collection name.  Defaults to ``"kitai"``.
        persist_directory (str | None): If given, the collection is written to
            disk at this path (auto-persisted on chromadb ≥ 0.4).  ``None``
            creates an ephemeral in-memory store.

    Returns:
        Chroma: Populated Chroma vector store ready for similarity search and
            ``SelfQueryRetriever``.

    Raises:
        ValueError: If ``docs`` is empty, lengths mismatch, ``metadata["id"]``
            is missing on any document, IDs are not unique, or
            ``collection_name`` is blank.
        KeyError: If any document is missing ``metadata["id"]``.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")
    if len(docs) != embeddings.shape[0]:
        raise ValueError(
            f"docs and embeddings must have the same length. "
            f"Got {len(docs)} docs and {embeddings.shape[0]} embedding rows."
        )
    if not collection_name:
        raise ValueError("collection_name must be a non-empty string.")

    # Validate ids — KeyError propagates if metadata["id"] is missing.
    ids = [str(doc.metadata["id"]) for doc in docs]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(
            f"metadata['id'] values must be unique. duplicate ids: {duplicates}"
        )

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    # chromadb Collection.upsert accepts float32 ndarray directly.
    emb_array = embeddings.astype(np.float32)

    logger.info(
        "Building Chroma store from pre-computed embeddings: %d docs, dim=%d, persist=%r",
        len(docs),
        embeddings.shape[1],
        persist_directory,
    )
    # Create an empty Chroma collection with query_encoder stored for query-time
    # use, then inject pre-computed vectors via Collection.upsert — which accepts
    # np.ndarray directly and does NOT call embedding_function during construction.
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=query_encoder,
        persist_directory=persist_directory,
    )
    vs._collection.upsert(
        ids=ids,
        embeddings=emb_array,
        documents=texts,
        metadatas=metadatas,
    )
    return vs


def save_chroma_vectorstore(
    vs: Chroma,
    persist_directory: str,
) -> Chroma:
    """
    Persist an existing in-memory Chroma vector store to disk.

    Reads all embeddings, documents, and metadata from ``vs`` via
    ``Collection.get()``, then writes them into a new Chroma instance backed
    by ``persist_directory``.  The embedding function and collection name are
    derived from ``vs`` — no second call to the embedding API is made.

    On chromadb >= 0.4 persistence is automatic once ``persist_directory`` is
    set — no separate ``.persist()`` call is required.

    Typical usage::

        vs = create_chroma_vectorstore(docs, embedding_fn, collection_name="my_col")
        persisted = save_chroma_vectorstore(vs, persist_directory="/path/to/db")
        # later:
        reloaded = load_chroma_vectorstore("/path/to/db", embedding_fn, collection_name="my_col")

    Invariants:
        - ``persist_directory`` must be a non-empty string.
        - ``vs._collection`` must be readable via ``Collection.get()``.

    Args:
        vs (Chroma): An existing Chroma vector store, typically created with
            :func:`create_chroma_vectorstore` or
            :func:`create_chroma_vectorstore_from_embeddings`.
        persist_directory (str): Local path where Chroma writes its SQLite
            database and segment files.

    Returns:
        Chroma: New Chroma instance backed by ``persist_directory``, with the
            same collection name and embedding function as ``vs``.

    Raises:
        ValueError: If ``persist_directory`` is blank.
    """
    if not persist_directory:
        raise ValueError("persist_directory must be a non-empty string.")

    collection_name = vs._collection.name
    data = vs._collection.get(include=["embeddings", "documents", "metadatas"])

    logger.info(
        "Persisting Chroma collection '%s' (%d docs) to '%s'",
        collection_name,
        len(data["ids"]),
        persist_directory,
    )

    persisted = Chroma(
        collection_name=collection_name,
        embedding_function=vs._embedding_function,
        persist_directory=persist_directory,
    )
    persisted._collection.upsert(
        ids=data["ids"],
        embeddings=data["embeddings"],
        documents=data["documents"],
        metadatas=data["metadatas"],
    )
    return persisted


def load_chroma_vectorstore(
    persist_directory: str,
    embedding_fn,
    collection_name: str = "kitai",
) -> Chroma:
    """
    Load a previously persisted Chroma vector store from disk.

    Args:
        persist_directory (str): Path passed to ``save_chroma_vectorstore``
            when the store was created.
        embedding_fn: The same LangChain ``Embeddings`` instance used at
            save time — must produce vectors of identical dimension.
        collection_name (str): Must match the name used at save time.
            Defaults to ``"kitai"``.

    Returns:
        Chroma: Loaded vector store ready for similarity search and
            ``SelfQueryRetriever``.

    Raises:
        ValueError: If ``persist_directory`` is blank.
        FileNotFoundError: If ``persist_directory`` does not exist on disk.
    """
    if not persist_directory:
        raise ValueError("persist_directory must be a non-empty string.")

    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: '{persist_directory}'. "
            "Did you call save_chroma_vectorstore first?"
        )

    logger.info("Loading Chroma vector store from '%s'", persist_directory)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn,
        collection_name=collection_name,
    )


# Backward-compat re-export — canonical definition moved to kitai.retriever.
from kitai.retriever import create_BM25retriever_from_docs  # noqa: F401
