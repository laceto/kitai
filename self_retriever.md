Use Chroma.from_embeddings or add_embeddings to load pre-computed vectors directly into Chroma without re-computing embeddings.

Chroma accepts lists of texts, embeddings, and metadatas.

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # For consistency, even if not used

# Your pre-computed data (adjust shapes)
texts = ["Pasta Al Forno Con I Funghi...", "Besciamella..."]  # page_content from docs
precomputed_embeddings = [  # List of lists/floats, shape (num_docs, embedding_dim)
    [0.1, 0.2, ...],  # 1536-dim for OpenAI, etc.
    [0.3, 0.4, ...]
]
metadatas = [  # From your docs.metadata
    {"id": 521, "recipe_id": "primi-dell-entroterra::Pasta Al Forno Con I Funghi", "title": "Pasta Al Forno Con I Funghi", "source_file": "primi-dell-entroterra"},
    {"id": 356, "recipe_id": "preparazioni-di-base::Besciamella", "title": "Besciamella", "source_file": "preparazioni-di-base"}
]
ids = ["doc_1", "doc_2"]  # Optional unique IDs

# Method 1: from_embeddings (builds new collection)
chroma = Chroma.from_embeddings(
    texts=texts,
    embeddings=precomputed_embeddings,  # Your pre-computed vectors
    metadatas=metadatas,
    ids=ids,
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()  # Dummy; not used for search since pre-computed
)

# Method 2: Load existing + add more
chroma = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
chroma.add_embeddings(
    texts=["new doc"],
    embeddings=[[0.1, 0.2, ...]],
    metadatas=[{"key": "value"}],
    ids=["new_id"]
)
chroma.persist()

# Use with self-query retriever
retriever = create_self_query_retriever(model, chroma, ...)

Embeddings must match your model's dimension (e.g., 1536 for text-embedding-3-small).
embedding_function is for new docs only; pre-computed bypasses it.
Self-query filters on metadatas; ensure they're complete.