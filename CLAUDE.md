# CLAUDE.md — kitai agent router

You are an agent working in the **kitai** Python library (RAG / embedding
utilities for LangChain-based pipelines). This file routes you to the right
context. Load only the file that matches your task.

---

## What is your task?

**Coding** — adding features, fixing bugs, refactoring source code, updating
notebooks or scripts
→ READ: `coding-rules.md`

**Testing** — writing new tests or expanding coverage
→ READ: `test-rules.md`

**Tests failing** — making red tests green
→ READ: `debugging-guide.md`, section "Test failures"
→ IGNORE: `test-rules.md`

**Debugging** — investigating runtime errors, unexpected results, import
issues, or batch job problems
→ READ: `debugging-guide.md`

---

## Load your rules now

1. Identify your task above.
2. Load **only** the corresponding file.
3. Do not load other files unless the rule file explicitly tells you to.
4. Do not guess — if your task spans multiple categories, ask for clarification.
