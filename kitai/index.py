from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import logging
from pathlib import Path
import os
from openai import OpenAI
import json
from typing import List, Tuple
import ast
# from typing import Optional
import numpy as np
import pandas as pd

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

def create_vectorstore(docs, embeddings, fake_embeddings_model):

    # # Create a FAISS index for the embedding dimension
    embedding_dim = get_embedding_dim(embeddings)
    # print(embedding_dim)
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

    # Add embeddings to the index
    index.add(embeddings)

    # Create an in-memory docstore mapping from internal index to document ID
    index_to_docstore_id = {i: doc.metadata["id"] for i, doc in enumerate(docs)}

    # Create the docstore with documents keyed by their IDs
    docstore = InMemoryDocstore({doc.metadata["id"]: doc for doc in docs})

    vector_store = FAISS(
        embedding_function=fake_embeddings_model,  # Not used since embeddings are precomputed
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vector_store


# Configure logging once at application entry point
logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)

import numpy as np

def get_embedding_dim(embeddings: np.ndarray) -> int:
    """
    Extract embedding dimension from a numpy array.

    Args:
        embeddings (np.ndarray): A 2D numpy array of shape (n_samples, n_features).

    Returns:
        int: The embedding dimension (number of features).
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {embeddings.ndim}D array instead.")
    
    return embeddings.shape[1]

def load_embeddings_from_csv(
    path_to_csv: str = './book_embeddings.csv',
    embedding_column: str = 'embedding',
    use_fake_embeddings: bool = False
) -> np.ndarray:
    """
    Load embeddings from CSV or generate fake embeddings for testing.
    
    Args:
        path_to_csv: Path to CSV file containing embeddings
        embedding_column: Column name with embedding data
        use_fake_embeddings: If True, return FakeEmbeddings for development
        
    Returns:
        np.ndarray: Stacked embeddings array
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If embedding_column not in CSV
        ValueError: If embedding format is invalid
    """
    try:
        # Load CSV with explicit error handling
        df_embeddings = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path_to_csv}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")
    
    # Verify column exists
    if embedding_column not in df_embeddings.columns:
        raise KeyError(
            f"Column '{embedding_column}' not found. "
            f"Available columns: {list(df_embeddings.columns)}"
        )
    
    # Parse embeddings with validation
    def parse_embedding(x: str) -> np.ndarray:
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Invalid embedding format. Expected list or array string. "
                f"Got: {x[:50]}... Error: {str(e)}"
            )
    
    df_embeddings["embedding_array"] = df_embeddings[embedding_column].apply(
        parse_embedding
    )
    
    # Stack embeddings
    embeddings = df_embeddings['embedding_array']
    return np.stack(embeddings)





def retrieve_embeddings_batches(client: OpenAI, job_ids: List[str]) -> List[Tuple[str, List[float]]]:
    """
    Retrieves embeddings from completed batch jobs.

    Args:
        client (OpenAI): Initialized OpenAI client.
        job_ids (List[str]): List of batch job IDs.

    Returns:
        List[Tuple[str, List[float]]]: A list of tuples (custom_id, embedding).
    """
    output_files_ids = []
    for job_id in job_ids:
        try:
            batch_info = client.batches.retrieve(job_id)
            output_files_ids.append(batch_info.output_file_id)
        except Exception as e:
            print(f"Error retrieving batch job {job_id}: {e}")

    output_files = []
    for output_file_id in output_files_ids:
        try:
            file_content = client.files.content(output_file_id).text
            output_files.append(file_content)
            lines = file_content.split('\n')
            print(f"File {output_file_id} contains {len(lines)} lines.")
        except Exception as e:
            print(f"Error retrieving file content for {output_file_id}: {e}")

    embedding_results = []
    for file_content in output_files:
        for line in file_content.split('\n')[:-1]:  # Skip last empty line
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id')
                embedding = data['response']['body']['data'][0]['embedding']
                embedding_results.append((custom_id, embedding))
            except Exception as e:
                print(f"Error parsing line: {e}")

    return embedding_results

def create_batch_files_embeddings(
    docs: List,
    batch_size: int = 20_000,
    batch_file_name: str = "icd_codes_batch",
    output_dir: str = "./batch_files"
) -> None:
    """
    Split docs into batches and write JSONL files for embeddings requests.

    Args:
        docs (List): List of document-like objects with `metadata['id']` and `page_content`.
        batch_size (int): Number of docs per batch file.
        batch_file_name (str): Base name for batch files.
        output_dir (str): Directory to store batch files.

    Raises:
        ValueError: If docs is empty or batch_size <= 0.
        OSError: If file operations fail.
    """
    if not docs:
        raise ValueError("Docs list cannot be empty.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    num_files = (len(docs) + batch_size - 1) // batch_size
    logging.info("Creating %d batch files in '%s'", num_files, output_path)

    for num_file in range(num_files):
        batch_docs = docs[num_file * batch_size : (num_file + 1) * batch_size]
        output_file = output_path / f"{batch_file_name}_part{num_file}.jsonl"

        if output_file.exists():
            logging.debug("Removing existing file: %s", output_file)
            output_file.unlink()

        try:
            with output_file.open("w", encoding="utf-8") as file:
                for doc in batch_docs:
                    payload = {
                        "custom_id": f"custom_id_{doc.metadata['id']}",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "input": doc.page_content,
                            "model": "text-embedding-3-small",
                            "encoding_format": "float",
                            "dimensions": 1536,
                        },
                    }
                    file.write(json.dumps(payload) + "\n")
            logging.info("Batch file created: %s", output_file)
        except Exception as e:
            logging.error("Failed to write batch file '%s': %s", output_file, e)
            raise


def create_BM25retriever_from_docs(
    docs: list[Document], 
    k : int
    ):  

    try:  
        if not docs:  
            raise ValueError("The documents list cannot be empty.")  
        if k <= 0:  
            raise ValueError("k must be a positive integer.")  
  
        bm25_retriever = BM25Retriever.from_documents(docs)  
        bm25_retriever.k = k  
        return bm25_retriever  
    except Exception as e:  
        print(f"An error occurred while creating the BM25 retriever: {e}")  
        return None
    




def create_embeddings_batches(client: OpenAI, batch_folder: str, completion_window: str = "24h") -> List[dict]:
    """
    Creates batch files and submits batch jobs for embeddings.

    Args:
        client (OpenAI): Initialized OpenAI client.
        batch_folder (str): Path to the folder containing input files.
        completion_window (str): Time window for batch completion (default: "24h").

    Returns:
        List[dict]: A list of job creation responses.
    """
    if not os.path.isdir(batch_folder):
        raise ValueError(f"Invalid folder path: {batch_folder}")

    batch_input_files = []
    job_creations = []

    # Create batch files
    for file_name in os.listdir(batch_folder):
        file_path = os.path.join(batch_folder, file_name)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    batch_file = client.files.create(file=f, purpose="batch")
                    batch_input_files.append(batch_file)
            except Exception as e:
                print(f"Error creating batch file for {file_name}: {e}")

    # Create batch jobs
    batch_file_ids = [batch_file.id for batch_file in batch_input_files]
    for i, file_id in enumerate(batch_file_ids):
        try:
            job = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/embeddings",
                completion_window=completion_window,
                metadata={"description": f"part_{i}_icd_embeddings"}
            )
            job_creations.append(job)
        except Exception as e:
            print(f"Error creating batch job for file ID {file_id}: {e}")

    # WE can see here the jobs created, they start with validation
    for job in job_creations:
        print(job)

    # we extract the ids for the job to check the status
    job_ids = [job.id for job in job_creations]

    return job_ids