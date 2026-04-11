# Coding Rules — kitai

## Package layout

```
kitai/
  __init__.py            — empty (no re-exports at package level)
  query_translation.py   — LLM-backed query expansion strategies
  transform.py           — convert strings / DataFrames into Documents
  index.py               — FAISS vector store, BM25, embedding batch I/O
  retriever.py           — retriever strategy helpers
  batch.py               — OpenAI Batch API primitives + embedding workflow
  export.py              — write DataFrames to CSV / Excel
  paths.py               — folder and file-path utilities
  tests/
    test_processor.py
```

User-guide notebooks (in `notebooks/`):
- `notebooks/index_guide.ipynb`
- `notebooks/retriever_guide.ipynb`
- `notebooks/query_translation_guide.ipynb`
- `notebooks/batch_api_guide.ipynb`

Raw dev / one-off scripts (in `scripts/`):
- `scripts/icd_batch_embedding.ipynb`  ← raw dev notebook, not a guide
- `scripts/hybrid_rag.py`              ← end-to-end hybrid RAG demo

---

## Import paths

kitai depends on `langchain-classic` (PyPI: `langchain_classic>=1.0`).
Never import from the bare `langchain` namespace — that package may resolve
to the Anthropic Agent SDK in some environments.

| Symbol | Correct import |
|---|---|
| `SelfQueryRetriever` | `from langchain_classic.retrievers.self_query.base import SelfQueryRetriever` |
| `EnsembleRetriever` | `from langchain_classic.retrievers.ensemble import EnsembleRetriever` |
| `AttributeInfo` | `from langchain_classic.chains.query_constructor.schema import AttributeInfo` |
| `BM25Retriever` | `from langchain_community.retrievers import BM25Retriever` |
| `LongContextReorder` | `from langchain_community.document_transformers import LongContextReorder` |
| `Document` | `from langchain_core.documents import Document` |
| `VectorStore` | `from langchain_core.vectorstores.base import VectorStore` |
| `BaseChatModel` | `from langchain_core.language_models.chat_models import BaseChatModel` |

In user-facing code and notebooks, prefer the kitai wrapper functions
(e.g. `create_hybrid_retriever`) so callers never need these import paths directly.

---

## Module-level invariants

### kitai.index
- `len(docs)` must equal `embeddings.shape[0]` — the i-th embedding must
  correspond to the i-th document. A mismatch raises `ValueError` before
  any FAISS index is built.
- Every `doc.metadata["id"]` must be unique and set — it is used as the
  key in the in-memory docstore and in `index_to_docstore_id`.
- The `fake_embeddings_model` parameter is stored for query-time encoding;
  it is **not** called during vector store construction.

### kitai.retriever
- Every function returns its declared type or raises — no `None`/`[]`
  sentinels on failure.
- `ValueError` guards fire before any network call or index build.
- `create_BM25retriever_from_docs` is the canonical definition in
  `kitai/retriever.py`; `kitai/index.py` re-exports it for backward compatibility.
- No `print()` calls; module-level `logger = logging.getLogger(__name__)`
  is used throughout.

### kitai.batch
- No module-level side effects: no client initialisation, no `load_dotenv()`.
- All network failures propagate to callers; per-item parse errors are
  logged at ERROR and skipped (partial-failure tolerant).
- Every document passed to `build_embedding_tasks` must carry
  `doc.metadata["id"]` — used to build the `custom_id` field.
- `BatchJobNotCompleteError.status` can be inspected to distinguish
  in-flight jobs (retry) from terminal failures (abort).

---

## Logging convention

All kitai modules use `logging.getLogger(__name__)` and configure **no
handlers**. The one exception is `kitai/paths.py`, which calls
`logging.basicConfig` at import time as a script convenience.

For application or notebook use, configure logging before importing kitai:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
```

Set `level=logging.DEBUG` to see individual batch upload/job IDs and
poll ticks from `kitai.batch`.

---

## Public API by module

### kitai.query_translation
| Symbol | Type | Notes |
|---|---|---|
| `decompose_query(model, user_query, few_shot_examples)` | function | returns `list[list[DecomposedQuery]]` |
| `step_back_query(model, user_query, num_queries, few_shot_examples)` | function | returns `list[list[GeneralQuery]]` |
| `expand_query(model, user_query, few_shot_examples)` | function | returns `list[list[ParaphrasedQuery]]` |
| `read_user_queries_from_excel(path, query_col_index)` | function | returns `(list[list[str]], list[str])` |
| `format_few_shot_examples(few_shot_examples, label)` | helper | pure, no model call |
| `create_input_objects(input_strings, **common_fields)` | helper | pure, no model call |
| `DecomposedQuery`, `GeneralQuery`, `ParaphrasedQuery` | Pydantic models | structured output types |

### kitai.transform
| Symbol | Type | Notes |
|---|---|---|
| `list_to_docs(docs)` | function | `list[str]` → `list[Document]` |
| `df_to_docs(df, content_column, metadata_columns)` | function | `DataFrame` → `list[Document]` |
| `add_metadata_to_docs(docs, key, value)` | function | immutable — returns new list |
| `add_num_id_to_metadata(docs)` | function | adds `metadata["id_new"]` |
| `flatten_list_of_lists(nested_list)` | function | flattens one level |
| `extract_attribute_doc(doc, attribute)` | function | single doc |
| `extract_attribute_docs(docs, attribute)` | function | list of docs |

### kitai.index
| Symbol | Type | Notes |
|---|---|---|
| `create_faiss_vectorstore_from_embeddings(docs, embeddings, query_encoder)` | function | returns `FAISS`; `create_vectorstore` is a deprecated alias |
| `create_chroma_vectorstore(docs, embedding_fn, collection_name)` | function | returns ephemeral `Chroma` (callable encoder); use for `SelfQueryRetriever` |
| `create_chroma_vectorstore_from_embeddings(docs, embeddings, query_encoder, collection_name, persist_directory)` | function | returns `Chroma` from pre-computed `np.ndarray`; supports `SelfQueryRetriever` |
| `save_chroma_vectorstore(vs, persist_directory)` | function | persists an existing `Chroma` instance to disk |
| `load_chroma_vectorstore(persist_directory, embedding_fn, collection_name)` | function | loads `Chroma` from disk; raises `FileNotFoundError` if path absent |
| `embed_documents(docs, embedding_fn)` | function | synchronous embedding — returns `float32 ndarray` of shape `(n_docs, dim)`; raises `ValueError` if `docs` is empty |
| `get_embedding_dim(embeddings)` | function | `ndarray` → `int` |
| `load_embeddings_from_csv(path_to_csv, embedding_column)` | function | returns `ndarray` |
| `create_BM25retriever_from_docs(docs, k)` | re-export | canonical definition in `kitai.retriever` |

### kitai.retriever
| Symbol | Type | Notes |
|---|---|---|
| `create_retriever(vs, search_type, search_kwargs)` | function | returns `VectorStoreRetriever` |
| `create_BM25retriever_from_docs(docs, k)` | function | canonical definition |
| `create_BM25retriever_from_text(docs, k)` | function | `list[str]` → `BM25Retriever` |
| `create_hybrid_retriever(sparse_retriever, semantic_retriever, weights_sparse)` | function | returns `EnsembleRetriever` |
| `create_self_query_retriever(model, vector_store, document_content_description, metadata_field_info, verbose)` | function | returns `SelfQueryRetriever` |
| `reorder_docs(docs)` | function | `LongContextReorder` post-processing |

### kitai.batch
| Symbol | Type | Notes |
|---|---|---|
| `submit_batch_job(client, tasks, endpoint, completion_window, metadata)` | function | returns job ID string |
| `check_batch_job(client, batch_id)` | function | returns status dict |
| `download_batch_results(client, batch_id)` | function | raises `BatchJobNotCompleteError` if not done |
| `poll_until_complete(client, batch_ids, poll_interval)` | function | blocking; returns completed IDs |
| `build_embedding_tasks(docs, model, dimensions)` | function | returns `list[dict]` |
| `parse_embedding_results(results)` | function | returns `list[tuple[str, list[float]]]` |
| `BatchJobNotCompleteError` | exception | `.batch_id`, `.status` attributes |
| `DEFAULT_EMBEDDING_MODEL` | constant | `"text-embedding-3-small"` |
| `DEFAULT_EMBEDDING_DIMENSIONS` | constant | `1536` |
| `build_chat_tasks(items, system_prompt, model)` | function | `items` is `list[dict]` with `"id"` and `"content"` keys |
| `parse_chat_results(results, extractor_fn)` | function | applies `extractor_fn: str → T`; returns `list[tuple[str, T]]`; skips errors |
| `DEFAULT_CHAT_MODEL` | constant | `"gpt-4o-mini"` |

### kitai.export
| Symbol | Notes |
|---|---|
| `df_to_csv(df, path)` | logs success at INFO; re-raises on error |
| `df_to_excel(df, path, sheet)` | logs success at INFO; re-raises on error |

### kitai.paths
| Symbol | Notes |
|---|---|
| `check_and_create_folder(folder_path)` | returns `Path`; raises `ValueError` / `OSError` |
| `get_file_paths(path, file_pattern)` | recursive glob by extension suffix |

---

## When Done

- Do not write tests inline — tests live in `kitai/tests/`.
- Do not add `print()` calls — use `logger`.
- Update `CLAUDE.md` public API table if a function signature changes.
