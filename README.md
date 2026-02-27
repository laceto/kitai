# kitai

LangChain utilities for query translation and document transformation.

## Features

- **Query translation** — three retrieval strategies (decompose, step-back, expand) backed by structured LLM output via `PydanticToolsParser`
- **Document helpers** — convert strings, DataFrames, and metadata into LangChain `Document` objects
- **Vector store / BM25 indexing** — FAISS and BM25 retriever creation from document lists
- **Excel ingestion** — read query lists from multi-sheet Excel files

## Installation

```bash
pip install .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

### Query translation

All three functions accept a `BaseChatModel` and a list of query strings.
They return a nested list — one list of result objects per input query.

```python
from langchain_openai import ChatOpenAI
from query_translation import decompose_query, step_back_query, expand_query

model = ChatOpenAI(model="gpt-4o")
queries = ["How is the premium calculated in this reinsurance agreement?"]

# Break a query into focused sub-questions
decomposed = decompose_query(model, queries)

# Lift a query to a higher conceptual level (default: 3 sub-queries per input)
abstract = step_back_query(model, queries, num_queries=3)

# Generate paraphrased variants for broader retrieval recall
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
from query_translation import read_user_queries_from_excel

# Returns (list[list[str]], list[str]) — one list per sheet, plus sheet names
queries_per_sheet, sheet_names = read_user_queries_from_excel(
    "queries.xlsx",
    query_col_index=[0, 2],  # column 0 for sheet 1, column 2 for sheet 2
)
```

## License

MIT License