import logging

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def list_to_docs(
    docs: list[str]
) -> list[Document]:
    """
    Create a list of LangChain Document objects from a list of strings.

    Args:
        docs (list[str]): List of strings.

    Returns:
        list[Document]: List of LangChain Document objects.

    Raises:
        ValueError: If the input list is empty or contains non-string elements.
    """
    if not docs:
        raise ValueError("The input list 'docs' cannot be empty.")

    documents = []
    for string in docs:
        if not isinstance(string, str):
            raise ValueError("All elements in 'docs' must be strings.")
        documents.append(Document(page_content=string))
    return documents


def flatten_list_of_lists(
    nested_list: list
) -> list:
    """
    Flattens a list of lists into a single list.

    Args:
        nested_list: A list containing lists to be flattened.

    Returns:
        A single list containing all elements from the nested lists.

    Raises:
        TypeError: If the input is not a list of lists.
    """
    if not all(isinstance(i, list) for i in nested_list):
        raise TypeError("Input must be a list of lists.")
    return [item for sublist in nested_list for item in sublist]


def extract_attribute_doc(
    doc: Document,
    attribute: str
) -> str:
    """Extracts the specified attribute from a document object.

    Args:
        doc: LangChain Document object from which to extract the attribute.
        attribute: Name of the attribute to extract from the document.

    Returns:
        Value of the specified attribute.

    Raises:
        AttributeError: If the document does not have the specified attribute.
    """
    if not hasattr(doc, attribute):
        raise AttributeError(f"Document does not have the '{attribute}' attribute.")
    return getattr(doc, attribute)


def extract_attribute_docs(
    docs: list[Document],
    attribute
) -> list[str]:
    """Extracts the specified attribute from a list of document objects.

    Args:
        docs: list[Document]: List of LangChain Document objects.
        attribute: Name of the attribute to extract from the document(s).

    Returns:
        List containing the values of the specified attribute from each document.

    Raises:
        AttributeError: If a document does not have the specified attribute.
        TypeError: If the input is not a document object or a list, or if the
            attribute is not a string.
    """
    if isinstance(docs, list):
        if not all(isinstance(doc, object) for doc in docs):
            raise TypeError("All items in the list must be document objects.")
    elif not isinstance(docs, object):
        raise TypeError("Input must be a document object or a list of document objects.")

    if not isinstance(attribute, str):
        raise TypeError("Attribute name must be a string.")

    attribute_values = []
    for doc in (docs if isinstance(docs, list) else [docs]):
        attribute_values.append(extract_attribute_doc(doc, attribute))
    return attribute_values


def add_num_id_to_metadata(
    docs: list[Document],
) -> list[Document]:
    """Update metadata of LangChain Document objects with a sequential numeric id.

    Given a list of LangChain Documents, this function updates each Document's
    metadata in-place to include a numeric ``id_new`` field equal to the
    document's zero-based position in the list, then returns the same list.

    Args:
        docs (list[Document]): List of LangChain Document objects.

    Returns:
        list[Document]: The same list with ``metadata["id_new"]`` set on every doc.
    """
    for i, doc in enumerate(docs):
        # Copy existing metadata or create new dict if None
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Add or overwrite the 'id_new' field with a progressive integer
        metadata["id_new"] = i
        doc.metadata = metadata
    return docs


def add_metadata_to_docs(
    docs: list[Document],
    key: str,
    value: any
) -> list[Document]:
    """Update metadata of LangChain Document objects.

    Given a list of LangChain Documents, this function returns a new list where
    each Document's metadata is updated to include a specified key-value pair.

    Args:
        docs (list[Document]): List of LangChain Document objects.
        key (str): Key to be added to the metadata.
        value (any): Value associated with the key to be added to the metadata.

    Returns:
        list[Document]: List of LangChain Document objects with updated metadata.
    """
    new_docs = []
    try:
        for doc in docs:
            new_metadata = dict(doc.metadata or {})
            new_metadata[key] = value
            new_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
        return new_docs

    except Exception as e:
        logger.error("An error occurred while adding metadata: %s", e)
        return []


def df_to_docs(
    df: pd.DataFrame,
    content_column: str,
    metadata_columns: list[str] = None
) -> list[Document]:
    """
    Prepare a list of Document objects from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        content_column (str): The name of the column containing document content.
        metadata_columns (list[str], optional): Columns to use as metadata.

    Returns:
        list[Document]: A list of Document objects created from the DataFrame rows.

    Raises:
        ValueError: If the content_column is not found in the DataFrame.
    """
    if content_column not in df.columns:
        raise ValueError(f"Content column '{content_column}' not found in the DataFrame")

    documents = []
    for _, row in df.iterrows():
        page_content = str(row[content_column])

        metadata = {}
        if metadata_columns:
            for col in metadata_columns:
                if col in df.columns:
                    metadata[col] = str(row[col]) if pd.notna(row[col]) else "Unknown"

        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents
