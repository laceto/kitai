# kitai — API dependency map

For each public function, shows whether it uses **LangChain** (accepts/returns
LangChain types or calls the LangChain chain machinery) and/or **OpenAI**
(calls the OpenAI REST API directly via the `openai` Python client).

**Legend**

| Symbol | Meaning |
|---|---|
| `LC` | Uses LangChain types or chain machinery |
| `OAI` | Calls OpenAI API directly (`openai` client) |
| `LC + OAI` | Uses both |
| `—` | Pure Python / pandas / numpy — no AI framework |

---

## `kitai.query_translation`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `decompose_query` | ✓ | | Invokes a LangChain chain: `prompt \| BaseChatModel \| parser` |
| `step_back_query` | ✓ | | Same chain pattern |
| `expand_query` | ✓ | | Same chain pattern |
| `create_input_objects` | | | Pure helper — builds plain dicts |
| `format_few_shot_examples` | | | Pure helper — formats strings |
| `read_user_queries_from_excel` | | | pandas only |

> The three query functions accept any `BaseChatModel` — the caller chooses
> the provider (OpenAI, Anthropic, etc.).  No direct OpenAI dependency here.

---

## `kitai.transform`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `list_to_docs` | ✓ | | Returns `list[Document]` |
| `df_to_docs` | ✓ | | Returns `list[Document]` |
| `add_metadata_to_docs` | ✓ | | Operates on `list[Document]` |
| `add_num_id_to_metadata` | ✓ | | Operates on `list[Document]` |
| `extract_attribute_doc` | ✓ | | Accepts `Document` |
| `extract_attribute_docs` | ✓ | | Accepts `list[Document]` |
| `flatten_list_of_lists` | | | Pure Python |

---

## `kitai.index`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `create_faiss_vectorstore_from_embeddings` | ✓ | | Returns LangChain `FAISS` wrapper |
| `create_chroma_vectorstore` | ✓ | | Returns `Chroma`; accepts LangChain `Embeddings` callable |
| `create_chroma_vectorstore_from_embeddings` | ✓ | | Returns `Chroma` from pre-computed `ndarray` |
| `save_chroma_vectorstore` | ✓ | | Accepts and returns `Chroma` instance |
| `load_chroma_vectorstore` | ✓ | | Returns `Chroma` |
| `embed_documents(docs, embedding_fn)` | ✓ | | Accepts `list[Document]` + LangChain `Embeddings`; returns `float32 ndarray` |
| `get_embedding_dim` | | | Pure numpy |
| `load_embeddings_from_csv` | | | pandas / numpy only |

---

## `kitai.retriever`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `create_retriever` | ✓ | | Wraps `VectorStore.as_retriever()` |
| `create_BM25retriever_from_docs` | ✓ | | Returns `BM25Retriever` (langchain-community) |
| `create_BM25retriever_from_text` | ✓ | | Same — accepts `list[str]` |
| `create_hybrid_retriever` | ✓ | | Returns `EnsembleRetriever` |
| `create_self_query_retriever` | ✓ | | Returns `SelfQueryRetriever`; accepts `BaseChatModel` |
| `reorder_docs` | ✓ | | Applies `LongContextReorder` |

> All functions are LangChain-only.  The LLM passed to `create_self_query_retriever`
> can be any `BaseChatModel` — no OpenAI hard dependency.

---

## `kitai.batch`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `submit_batch_job` | | ✓ | Calls `client.batches.create()` |
| `check_batch_job` | | ✓ | Calls `client.batches.retrieve()` |
| `download_batch_results` | | ✓ | Calls `client.files.content()` |
| `poll_until_complete` | | ✓ | Polling loop over `check_batch_job` |
| `build_embedding_tasks` | ✓ | | Accepts `list[Document]`; returns OpenAI Batch API JSONL dicts |
| `build_chat_tasks` | | | Accepts plain dicts; returns OpenAI Batch API JSONL dicts |
| `parse_embedding_results` | | | Pure Python — parses list of response dicts |
| `parse_chat_results` | | | Pure Python — applies caller-supplied extractor |

> `build_embedding_tasks` is the only function that bridges both worlds:
> it reads `doc.metadata["id"]` (LangChain `Document`) and produces
> OpenAI Batch API payloads.

---

## `kitai.export`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `df_to_csv` | | | pandas only |
| `df_to_excel` | | | pandas only |

---

## `kitai.paths`

| Function | LC | OAI | Notes |
|---|:---:|:---:|---|
| `check_and_create_folder` | | | stdlib only |
| `get_file_paths` | | | stdlib only |

---

## Summary by module

| Module | LangChain | OpenAI direct |
|---|:---:|:---:|
| `query_translation` | ✓ | |
| `transform` | ✓ | |
| `index` | ✓ | |
| `retriever` | ✓ | |
| `batch` | partial (`build_embedding_tasks`) | ✓ |
| `export` | | |
| `paths` | | |
