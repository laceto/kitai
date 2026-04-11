# Test Rules — kitai

## TDD cycle (mandatory)

1. **Red** — write the test first; confirm it fails before writing any implementation.
2. **Green** — write the minimum code to make the test pass, nothing more.
3. **Refactor** — clean up with the safety net in place.

Never write implementation code without a failing test that demands it.

---

## Test location and structure

- Tests live in `kitai/tests/`.
- One test file per module (e.g. `test_index.py` for `kitai/index.py`).
- One `describe`-equivalent class or section per public function.
- Test names: `test_<function>_<scenario>` (e.g. `test_embed_documents_raises_on_empty`).
- Pattern: **Arrange → Act → Assert**.

---

## Coverage requirements

Every test must cover:
- **Happy path** — primary intended behaviour.
- **Edge cases** — empty inputs, boundary values, single-element lists.
- **Failure modes** — expected exceptions with correct type and message.

Add **characterisation tests** before refactoring any unclear or legacy behaviour.

---

## No implementation changes while testing

- Do NOT modify source code to make a test pass by bending the contract.
- If you discover a bug → create a task, do not fix it here.
- If tests reveal a missing feature → document it, do not implement it now.

---

## Import paths in tests

Use kitai's wrapper functions, not raw LangChain classes:

```python
# Good
from kitai.retriever import create_hybrid_retriever

# Bad — bare `langchain` may resolve to the Anthropic SDK
from langchain.retrievers import EnsembleRetriever
```

See `coding-rules.md` for the full venv import table if raw classes are needed.

---

## When Done

- Run: `python -m pytest kitai/tests/ -v`
- Report pass/fail count.
- If ALL pass → STOP.
- If ANY fail → LOAD: `debugging-guide.md`, section "Test failures".
