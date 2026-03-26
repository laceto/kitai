# kitai

LangChain utilities for RAG pipelines: query translation, document transformation,
FAISS vector-store creation, BM25/hybrid retrieval, and OpenAI Batch API embedding
workflows.

## Package layout

```
kitai/
  query_translation.py  — decompose / step-back / expand queries via LLM
  transform.py          — convert strings, DataFrames, and metadata into Documents
  index.py              — FAISS vector store, BM25 retriever, embedding CSV helpers
  retriever.py          — retriever strategy helpers (vector, BM25, hybrid, self-query, reorder)
  batch.py              — OpenAI Batch API primitives + embedding workflow helpers
  export.py             — write DataFrames to CSV / Excel
  paths.py              — folder creation and file-path listing utilities

notebooks/
  index_guide.ipynb             — vector store & BM25 user guide
  retriever_guide.ipynb         — retrieval strategies user guide
  query_translation_guide.ipynb — query translation user guide
  batch_api_guide.ipynb         — OpenAI Batch API user guide

scripts/
  hybrid_rag.py                 — end-to-end hybrid RAG demo
  icd_batch_embedding.ipynb     — raw dev notebook (not a guide)
```

## Installation

```bash
pip install .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

### Query translation (`kitai.query_translation`)

All three functions accept a `BaseChatModel` and a list of query strings.
They return a nested list — one list of result objects per input query.

```python
from langchain_openai import ChatOpenAI
from kitai.query_translation import decompose_query, step_back_query, expand_query

model = ChatOpenAI(model="gpt-4o")
queries = ["How is the premium calculated in this reinsurance agreement?"]

# Break a query into focused sub-questions → list[list[DecomposedQuery]]
decomposed = decompose_query(model, queries)

# Lift a query to a higher conceptual level → list[list[GeneralQuery]]
abstract = step_back_query(model, queries, num_queries=3)

# Generate paraphrased variants for broader retrieval recall → list[list[ParaphrasedQuery]]
expanded = expand_query(model, queries)
```

**Injecting domain-specific few-shot examples:**

```python
custom_examples = [
    {
        "original_query": "What is the effective date of this policy?",
        "new_queries": [
            "When does this policy come into force?",
            "What is the start date specified in the agreement?",
        ],
    }
]
expanded = expand_query(model, queries, few_shot_examples=custom_examples)
```

**Reading queries from Excel:**

```python
from kitai.query_translation import read_user_queries_from_excel

# Returns (list[list[str]], list[str]) — one list per sheet, plus sheet names
queries_per_sheet, sheet_names = read_user_queries_from_excel(
    "queries.xlsx",
    query_col_index=[0, 2],  # column 0 for sheet 1, column 2 for sheet 2
)
```

---

### Document transformation (`kitai.transform`)

Convert raw data into LangChain `Document` objects.

```python
from kitai.transform import list_to_docs, df_to_docs, add_metadata_to_docs, add_num_id_to_metadata

# From plain strings
docs = list_to_docs(["paragraph one", "paragraph two"])

# From a DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
docs = df_to_docs(df, content_column="text", metadata_columns=["source", "date"])

# Attach a constant metadata field to every document
docs = add_metadata_to_docs(docs, key="source", value="contract_v1")

# Add a sequential integer id to each document's metadata
docs = add_num_id_to_metadata(docs)
```

**Utility helpers:**

```python
from kitai.transform import flatten_list_of_lists, extract_attribute_docs

# Flatten list-of-lists returned by query translation functions
all_queries = flatten_list_of_lists(decomposed)

# Extract a single attribute from every Document
texts = extract_attribute_docs(docs, "page_content")
```

---

### Vector store & BM25 indexing (`kitai.index`)

Build a FAISS or Chroma vector store from documents, or a BM25 retriever.

```python
import numpy as np
from kitai.index import (
    create_faiss_vectorstore_from_embeddings,
    load_embeddings_from_csv,
    create_BM25retriever_from_docs,
)

# Load pre-computed embeddings from CSV
embeddings: np.ndarray = load_embeddings_from_csv(
    path_to_csv="embeddings.csv",
    embedding_column="embedding",
)

# Build FAISS vector store (pre-computed embeddings required)
# docs[i].metadata["id"] must be set — used as the docstore key
vector_store = create_faiss_vectorstore_from_embeddings(docs, embeddings, query_encoder)

# BM25 from Documents (re-exported in kitai.retriever too)
bm25 = create_BM25retriever_from_docs(docs, k=5)
```

> **Invariant:** `len(docs)` must equal `embeddings.shape[0]` and every
> document must carry `metadata["id"]`.
> `create_vectorstore` is a deprecated alias for `create_faiss_vectorstore_from_embeddings`.

**Chroma vector store** — required for `SelfQueryRetriever` (FAISS does not
support structured metadata filtering):

```python
from langchain_openai import OpenAIEmbeddings
from kitai.index import (
    create_chroma_vectorstore,                 # ephemeral, from callable encoder
    create_chroma_vectorstore_from_embeddings, # ephemeral or persistent, from np.ndarray
    save_chroma_vectorstore,                   # persistent, from callable encoder
    load_chroma_vectorstore,                   # load from disk
)

embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")

# Callable encoder path — Chroma calls embed_documents at index time
chroma_vs = create_chroma_vectorstore(docs, embedding_fn)

# Pre-computed embeddings path — same ndarray workflow as FAISS, no re-embedding
chroma_vs = create_chroma_vectorstore_from_embeddings(
    docs, embeddings, query_encoder=embedding_fn
)

# Persist to disk (auto-persisted on chromadb ≥ 0.4)
chroma_vs = save_chroma_vectorstore(docs, embedding_fn, persist_directory="./chroma_db")

# Reload from a previously saved directory
chroma_vs = load_chroma_vectorstore("./chroma_db", embedding_fn)
```

---

### Retrieval strategies (`kitai.retriever`)

```python
from kitai.retriever import (
    create_retriever,
    create_BM25retriever_from_docs,
    create_BM25retriever_from_text,
    create_hybrid_retriever,
    create_self_query_retriever,
    reorder_docs,
)

# Vector-similarity or MMR retriever from a FAISS vector store
retriever = create_retriever(
    vector_store,
    search_type="mmr",           # "similarity" | "similarity_score_threshold" | "mmr"
    search_kwargs={"k": 6},
)

# Sparse BM25 from plain strings
bm25 = create_BM25retriever_from_text(["doc text one", "doc text two"], k=4)

# Hybrid: combine sparse + semantic with explicit weights
hybrid = create_hybrid_retriever(
    sparse_retriever=bm25,
    semantic_retriever=retriever,
    weights_sparse=0.4,          # semantic gets 1 - 0.4 = 0.6
)

# Self-query retriever — LLM generates a metadata filter automatically
# Requires a Chroma vector store (FAISS does not support structured filters)
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from kitai.index import create_chroma_vectorstore

chroma_store = create_chroma_vectorstore(docs, embedding_fn)

metadata_field_info = [
    AttributeInfo(name="year", description="Publication year", type="integer"),
    AttributeInfo(name="author", description="Document author", type="string"),
]
sq_retriever = create_self_query_retriever(
    model,
    chroma_store,
    document_content_description="Insurance contract clauses",
    metadata_field_info=metadata_field_info,
)

# LongContextReorder — move most-relevant docs away from the middle
reordered = reorder_docs(retrieved_docs)
```

---

### OpenAI Batch API (`kitai.batch`)

Three independent layers — generic primitives, embedding-specific helpers,
and chat-completion helpers.

#### Generic primitives

```python
from kitai.batch import submit_batch_job, check_batch_job, download_batch_results

# Build task dicts yourself for any endpoint, then submit
job_id = submit_batch_job(
    client,
    tasks=my_task_list,
    endpoint="/v1/embeddings",
    metadata={"description": "my_run"},
)

# Poll a single job
status = check_batch_job(client, job_id)
# status keys: batch_id, status, is_terminal, is_complete, counts, output_file_id, error_file_id

# Download — raises BatchJobNotCompleteError if not yet done
results = download_batch_results(client, job_id)
```

#### Embedding workflow

```python
from kitai.batch import (
    build_embedding_tasks,
    poll_until_complete,
    parse_embedding_results,
    submit_batch_job,
    download_batch_results,
)

# 1. Build tasks (docs must have metadata["id"])
tasks = build_embedding_tasks(docs)

# 2. Submit (one job per batch; split docs yourself if needed)
job_id = submit_batch_job(client, tasks, metadata={"description": "embed_run_1"})

# 3. Block until all jobs finish; returns only the completed ones
completed_ids = poll_until_complete(client, [job_id], poll_interval=10.0)

# 4. Download raw results
results = download_batch_results(client, completed_ids[0])

# 5. Extract (custom_id, embedding) pairs
pairs = parse_embedding_results(results)
# pairs[i] == ("custom_id_<doc.metadata['id']>", [0.12, -0.03, ...])
```

#### Chat completion workflow

```python
from kitai.batch import (
    build_chat_tasks,
    poll_until_complete,
    parse_chat_results,
    submit_batch_job,
    download_batch_results,
)

# 1. Build tasks — each item needs "id" and "content" keys
items = [{"id": "1", "content": "Summarise this clause..."}, ...]
tasks = build_chat_tasks(items, system_prompt="You are a concise summariser.")

# 2. Submit
job_id = submit_batch_job(
    client, tasks,
    endpoint="/v1/chat/completions",
    metadata={"description": "summarise_run_1"},
)

# 3. Wait for completion
completed_ids = poll_until_complete(client, [job_id], poll_interval=10.0)

# 4. Download raw results
results = download_batch_results(client, completed_ids[0])

# 5. Extract (custom_id, value) pairs — supply any extractor for the response text
pairs = parse_chat_results(results, extractor_fn=str.strip)
# or with a JSON extractor:
# pairs = parse_chat_results(results, extractor_fn=json.loads)
```

---

### Export helpers (`kitai.export`)

```python
from kitai.export import df_to_csv, df_to_excel

df_to_csv(df, "output/results.csv")
df_to_excel(df, "output/results.xlsx", sheet="Results")
```

---

### Path utilities (`kitai.paths`)

```python
from kitai.paths import check_and_create_folder, get_file_paths

# Ensure a folder exists (creates it if necessary)
folder = check_and_create_folder("./output/batch_files")

# Recursively collect files by extension
pdf_paths = get_file_paths("./documents", file_pattern=".pdf")
```

---

## Logging

All modules use `logging.getLogger(__name__)`.  No module configures handlers.
To see debug output, configure logging at application entry:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
```

`kitai/paths.py` calls `logging.basicConfig` at import time as a convenience
for scripts; applications should configure logging before importing this module
if custom handlers are required.

---

## License

MIT License
