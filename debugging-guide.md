# Debugging Guide — kitai

## FAISS vector store returns unexpected results

1. Confirm `len(docs) == embeddings.shape[0]`.
2. Confirm every `doc.metadata["id"]` is unique and set.
3. With `FakeEmbeddings`, rankings are random — results are only meaningful
   with a real embedding model (`OpenAIEmbeddings`, etc.).
4. Use `similarity_search_by_vector(vec, k)` to bypass the query encoder
   and test with a known embedding.

---

## Batch job stuck in `"validating"`

OpenAI validates the uploaded JSONL file before processing. Wait 1–2 minutes,
then re-check with `check_batch_job`. If still stuck, inspect the file
content: every line must be valid JSON matching the Batch API schema.

---

## Batch job shows `status == "failed"`

```python
status = check_batch_job(client, batch_id)
error_text = client.files.content(status["error_file_id"]).text
print(error_text)
```

`BatchJobNotCompleteError.status` distinguishes in-flight jobs (retry) from
terminal failures (abort).

---

## `parse_embedding_results` skips items

Set `logging.DEBUG` before calling — each skipped item logs its `custom_id`
and the exact error. Common causes: wrong response shape, item-level
`"error"` field set by the API.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Self-query retriever returns empty results

Enable `verbose=True` in `create_self_query_retriever`. The LLM's generated
structured query (filter + semantic query string) will be logged. Check that:

1. The filter field names match the `name` field in `AttributeInfo` exactly.
2. The vector store backend supports structured query filtering — FAISS does
   **not** in this version. Use `kitai.index.create_chroma_vectorstore` to
   build a compatible Chroma store (`chromadb` must be installed).

---

## ImportError from LangChain symbols

kitai imports `SelfQueryRetriever`, `EnsembleRetriever`, and `AttributeInfo`
from `langchain_classic` (PyPI: `langchain-classic>=1.0`). If these raise
`ImportError`, the package is missing:

```bash
pip install 'langchain_classic>=1.0'
```

Do NOT import from the bare `langchain` namespace — it may resolve to the
Anthropic Agent SDK in some environments, which does not export these symbols.
See the full import table in `coding-rules.md`.

---

## Test failures

1. Read the failing test name and assertion carefully.
2. Trace through the source module — do NOT modify the test.
3. Find the mismatch between what the implementation returns and what the
   test expects.
4. Make the minimal source change to satisfy the test contract.
5. Re-run: `python -m pytest kitai/tests/ -v`
6. If all pass → STOP. Do not refactor while fixing.

---

## When Done

- Document the root cause in the relevant commit message or task.
- If the fix is non-trivial, update the appropriate section of this file.
