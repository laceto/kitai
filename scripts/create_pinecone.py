from dotenv import load_dotenv 
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)

from pinecone import Pinecone, ServerlessSpec

load_dotenv()
embeddings = OpenAIEmbeddings()



index_name = "langchain-pinecone-hybrid-search"

# initialize Pinecone client
pc = Pinecone()

# create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

from pinecone_text.sparse import BM25Encoder
bm25_encoder = BM25Encoder().default()

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)
retriever.add_texts(texts=extract_attribute_docs(docs, 'page_content'))
result = retriever.invoke(query)

rich.print(result)
