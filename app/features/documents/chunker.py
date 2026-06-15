"""Markdown generator + smart chunker.

Direct port from the notebook (Cell 4).
Converts loaded documents to markdown, then splits into chunks
that respect page boundaries and preserve tables as atomic units.
"""

import re
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_MARKER_RE = re.compile(r'<!--\s*page:(\d+)\s*-->')


def generate_markdown(docs: List[Document], file_name: str) -> str:
    """Convert LangChain Documents to a single markdown string with page markers."""
    parts = []
    for doc in docs:
        page_num = doc.metadata.get('page_number', 1)
        text = doc.page_content.strip()
        if text:
            parts.append(f'<!-- page:{page_num} -->\n\n{text}')
    return '\n\n---\n\n'.join(parts)


def smart_chunk(markdown_text: str, chunk_size: int = 500,
                chunk_overlap: int = 50) -> List[Dict]:
    """Split markdown into chunks, preserving page boundaries and tables."""
    chunks: List[Dict] = []
    segments = PAGE_MARKER_RE.split(markdown_text)

    # Handle text before any page marker
    if segments[0].strip():
        _chunk_section(segments[0], 1, chunk_size, chunk_overlap, chunks)

    i = 1
    while i + 1 <= len(segments) - 1:
        try:
            page_number = int(segments[i])
        except (ValueError, IndexError):
            page_number = 1
        content = segments[i + 1]
        if content.strip():
            _chunk_section(content, page_number, chunk_size, chunk_overlap, chunks)
        i += 2

    return [c for c in chunks if c['text'].strip()]


def _chunk_section(text: str, page_number: int, chunk_size: int,
                   chunk_overlap: int, result: list):
    """Split a page section into prose chunks and atomic table chunks."""
    lines = text.split('\n')
    prose_buf: List[str] = []
    table_buf: List[str] = []
    in_table = False

    def flush_prose():
        if not prose_buf:
            return
        prose_text = '\n'.join(prose_buf).strip()
        prose_buf.clear()
        if not prose_text:
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )
        for ct in splitter.split_text(prose_text):
            if ct.strip():
                result.append({'text': ct.strip(), 'page_number': page_number, 'is_table': False})

    def flush_table():
        if not table_buf:
            return
        table_text = '\n'.join(table_buf).strip()
        table_buf.clear()
        if table_text:
            result.append({'text': table_text, 'page_number': page_number, 'is_table': True})

    for line in lines:
        is_table_row = bool(re.match(r'\s*\|', line)) or (
            bool(re.match(r'^\s*[-:| ]+$', line)) and '|' in line
        )
        if is_table_row:
            if not in_table:
                flush_prose()
                in_table = True
            table_buf.append(line)
        else:
            if in_table:
                flush_table()
                in_table = False
            prose_buf.append(line)

    flush_table() if in_table else flush_prose()
