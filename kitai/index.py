from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import logging
from pathlib import Path
import os
from openai import OpenAI
import json
from typing import List, Tuple
import ast
import numpy as np
import pandas as pd

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

logger = logging.getLogger(__name__)

def create_vectorstore(
    docs: list[Document],
    embeddings: np.ndarray,
    fake_embeddings_model,
) -> FAISS:
    """
    Build a FAISS vector store from pre-computed embeddings and documents.

    Invariants:
        - ``len(docs)`` must equal ``embeddings.shape[0]``.  A mismatch means
          the i-th embedding would be paired with the wrong document.
        - Every document must have ``doc.metadata["id"]`` set; it is used as the
          key in the in-memory docstore and in ``index_to_docstore_id``.

    Args:
        docs (list[Document]): Documents to store.  Must be the same length as
            ``embeddings`` and each must carry a ``metadata["id"]`` field.
        embeddings (np.ndarray): 2-D float array of shape (n_docs, embedding_dim)
            with pre-computed embeddings in the same order as ``docs``.
        fake_embeddings_model: A LangChain embeddings object passed to FAISS as
            ``embedding_function``.  It is not called at construction time because
            the embeddings are pre-computed; it is only stored for future
            similarity-search calls that require re-encoding a query.

    Returns:
        FAISS: Populated FAISS vector store ready for similarity search.

    Raises:
        ValueError: If ``len(docs) != embeddings.shape[0]``.
    """
    if len(docs) != embeddings.shape[0]:
        raise ValueError(
            f"docs and embeddings must have the same length. "
            f"Got {len(docs)} docs and {embeddings.shape[0]} embedding rows."
        )

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
) -> np.ndarray:
    """
    Load embeddings from a CSV file.

    Args:
        path_to_csv: Path to CSV file containing embeddings.
        embedding_column: Column name with embedding data.

    Returns:
        np.ndarray: Stacked embeddings array of shape (n_rows, embedding_dim).

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        KeyError: If embedding_column not in CSV.
        ValueError: If embedding format is invalid.
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
    failed_jobs = []
    for job_id in job_ids:
        try:
            batch_info = client.batches.retrieve(job_id)
            output_files_ids.append(batch_info.output_file_id)
        except Exception as e:
            logger.error("Error retrieving batch job %s: %s", job_id, e)
            failed_jobs.append(job_id)

    output_files = []
    failed_files = []
    for output_file_id in output_files_ids:
        try:
            file_content = client.files.content(output_file_id).text
            output_files.append(file_content)
            lines = file_content.split('\n')
            logger.debug("File %s contains %d lines.", output_file_id, len(lines))
        except Exception as e:
            logger.error("Error retrieving file content for %s: %s", output_file_id, e)
            failed_files.append(output_file_id)

    embedding_results = []
    for file_content in output_files:
        for line in file_content.split('\n')[:-1]:  # Skip last empty line
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id')
                embedding = data['response']['body']['data'][0]['embedding']
                embedding_results.append((custom_id, embedding))
            except Exception as e:
                logger.error("Error parsing line: %s", e)

    logger.info(
        "Retrieved %d embeddings. Failed jobs: %d, failed files: %d",
        len(embedding_results),
        len(failed_jobs),
        len(failed_files),
    )
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
    logger.info("Creating %d batch files in '%s'", num_files, output_path)

    for num_file in range(num_files):
        batch_docs = docs[num_file * batch_size : (num_file + 1) * batch_size]
        output_file = output_path / f"{batch_file_name}_part{num_file}.jsonl"

        if output_file.exists():
            logger.debug("Removing existing file: %s", output_file)
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
            logger.info("Batch file created: %s", output_file)
        except Exception as e:
            logger.error("Failed to write batch file '%s': %s", output_file, e)
            raise


def create_BM25retriever_from_docs(
    docs: list[Document],
    k: int,
) -> BM25Retriever:
    """
    Create a BM25 retriever from a list of documents.

    Args:
        docs (list[Document]): Non-empty list of LangChain Document objects.
        k (int): Number of top documents to return per query.

    Returns:
        BM25Retriever: Configured BM25 retriever.

    Raises:
        ValueError: If docs is empty or k is not a positive integer.
    """
    if not docs:
        raise ValueError("docs must be a non-empty list.")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    return bm25_retriever
    




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
    failed_uploads = []
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
                logger.error("Error creating batch file for %s: %s", file_name, e)
                failed_uploads.append(file_name)

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
            logger.debug("Batch job created: %s", job)
        except Exception as e:
            logger.error("Error creating batch job for file ID %s: %s", file_id, e)

    logger.info(
        "Submitted %d batch jobs. %d files failed upload.",
        len(job_creations),
        len(failed_uploads),
    )

    # we extract the ids for the job to check the status
    job_ids = [job.id for job in job_creations]

    return job_ids