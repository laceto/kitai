import sys
import os
import pandas as pd
import rich

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kitai.paths import get_file_paths, check_and_create_folder
from kitai.transform import df_to_docs, add_num_id_to_metadata, add_metadata_to_docs
from kitai.index import create_BM25retriever_from_docs, create_batch_files_embeddings, retrieve_embeddings_batches
from kitai.transform import extract_attribute_docs
from kitai.export import df_to_excel, df_to_csv

from langchain_core.embeddings import FakeEmbeddings

folder = 'C:/Users/l_ace/Desktop/projects/rss_feed/output'
path = '.txt'
file_list = get_file_paths(folder, path)

dfs = [pd.read_csv(file, sep='\t') for file in file_list]
dfs = pd.concat(dfs, ignore_index=True)
dfs = dfs.drop_duplicates(subset='guid').reset_index(drop=True)
dfs['description'] = dfs['title'] + '. ' + dfs['description']
dfs = dfs.drop_duplicates(subset='title').reset_index(drop=True)
# df_to_excel(dfs, 'feeds.xlsx')

docs = df_to_docs(dfs, 'description', ['link', 'guid', 'type', 'sponsored', 'id', 'title', 'pubDate'])

bm25_retriever = create_BM25retriever_from_docs(docs=docs, k=5)
query = 'what president trump said about AI'
docs_retr = bm25_retriever.invoke(query)


batch_files_embeddings_folder = './data/batch_files/input'
check_and_create_folder(batch_files_embeddings_folder)

create_batch_files_embeddings(
    docs=docs,
    batch_file_name='rss_feeds',
    output_dir=batch_files_embeddings_folder
)

from kitai.index import create_embeddings_batches, load_embeddings_from_csv, get_embedding_dim, create_faiss_vectorstore_from_embeddings
from openai import OpenAI
from dotenv import load_dotenv 
load_dotenv()


# create batches embeddings and retrievial from openai
# client = OpenAI()
# job_ids = create_embeddings_batches(client= client, batch_folder=batch_files_embeddings_folder)
# job_ids = ['batch_691977fd05888190be2e4634e85ef152']
# embedding_results = retrieve_embeddings_batches(client, job_ids)
# embedding_results = pd.DataFrame(embedding_results, columns=['custom_id', 'embedding'])
# batch_files_embeddings_folder = './data/batch_files/output/'
# check_and_create_folder(batch_files_embeddings_folder)
# embedding_results.to_csv(batch_files_embeddings_folder+job_ids[0]+'.csv', index=False)

# create vectorstore with batches embeddings
# path_vectorstore = "./vectorstore/rss_feeds"
# check_and_create_folder(path_vectorstore)
# embeddings = load_embeddings_from_csv('./data/batch_files/output/batch_691977fd05888190be2e4634e85ef152.csv')
# embeddings_size = get_embedding_dim(embeddings)
# fake_embeddings_model = FakeEmbeddings(size = embeddings_size)
# vector_store = create_faiss_vectorstore_from_embeddings(docs, embeddings, fake_embeddings_model)
# vector_store.save_local(path_vectorstore, index_name='faiss_index_rss_feeds')

# vector_store = FAISS.load_local(path_vectorstore, index_name='faiss_index_rss_feeds', embeddings=fake_embeddings_model, allow_dangerous_deserialization=True)
# print(embeddings_size)

