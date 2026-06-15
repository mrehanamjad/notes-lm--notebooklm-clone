"""File loader — PDF, DOCX, MD, CSV, TXT.

Direct port from the notebook (Cell 3), adapted for server-side use.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from app.core.logger import logger

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.md', '.markdown', '.csv', '.txt'}


def is_supported(file_name: str) -> bool:
    """Check if the file extension is supported."""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(file_path: str, file_name: str) -> List[Document]:
    """Load a file into LangChain Document objects."""
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Unsupported file type: {ext}')

    docs = _load_by_ext(file_path, ext)

    for doc in docs:
        doc.metadata['file_name'] = file_name
        doc.metadata['file_type'] = ext.lstrip('.')

    logger.info(f"Loaded {len(docs)} page(s) from {file_name}")
    return docs


def _load_by_ext(file_path: str, ext: str) -> List[Document]:
    """Dispatch to the correct loader based on file extension."""
    if ext == '.pdf':
        docs = PyPDFLoader(file_path).load()
        for doc in docs:
            doc.metadata['page_number'] = doc.metadata.get('page', 0) + 1
        return docs

    if ext == '.docx':
        docs = Docx2txtLoader(file_path).load()
        for doc in docs:
            doc.metadata['page_number'] = 1
        return docs

    if ext in ('.md', '.markdown'):
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            docs = UnstructuredMarkdownLoader(file_path).load()
        except (ImportError, ModuleNotFoundError):
            docs = TextLoader(file_path, encoding='utf-8').load()
        for doc in docs:
            doc.metadata['page_number'] = 1
        return docs

    if ext == '.csv':
        docs = CSVLoader(file_path, encoding='utf-8').load()
        for i, doc in enumerate(docs):
            doc.metadata['page_number'] = i + 1
        return docs

    if ext == '.txt':
        docs = TextLoader(file_path, encoding='utf-8').load()
        for doc in docs:
            doc.metadata['page_number'] = 1
        return docs

    raise ValueError(f'No loader for: {ext}')
