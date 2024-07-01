import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
import os

from database_utils.db_catalog.csv_utils import load_tables_description

load_dotenv(override=True)

EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")

if os.environ.get("AZURE_OPENAI_API_KEY", None) is not None:
    EMBEDDING_FUNCTION = AzureOpenAIEmbeddings(
        openai_api_key=os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY", None),
        azure_endpoint=os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", None),
        azure_deployment=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", None),
        openai_api_version="2023-05-15",
    )


def make_db_context_vec_db(db_directory_path: str, **kwargs) -> None:
    """
    Creates a context vector database for the specified database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs: Additional keyword arguments, including:
            - use_value_description (bool): Whether to include value descriptions (default is True).
    """
    db_id = Path(db_directory_path).name

    table_description = load_tables_description(
        db_directory_path, kwargs.get("use_value_description", True))
    docs = []

    for table_name, columns in table_description.items():
        for column_name, column_info in columns.items():
            metadata = {
                "table_name": table_name,
                "original_column_name": column_name,
                "column_name": column_info.get('column_name', ''),
                "column_description": column_info.get('column_description', ''),
                "value_description": column_info.get('value_description', '') if kwargs.get("use_value_description", True) else ""
            }
            for key in ['column_name', 'column_description', 'value_description']:
                if column_info.get(key, '').strip():
                    docs.append(
                        Document(page_content=column_info[key], metadata=metadata))

    logging.info(f"Creating context vector database for {db_id}")
    vector_db_path = Path(db_directory_path) / "context_vector_db"

    if vector_db_path.exists():
        os.system(f"rm -r {vector_db_path}")

    vector_db_path.mkdir(exist_ok=True)

    Chroma.from_documents(docs, EMBEDDING_FUNCTION,
                          persist_directory=str(vector_db_path))

    logging.info(f"Context vector database created at {vector_db_path}")
