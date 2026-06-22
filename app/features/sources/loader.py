"""
File loader — PDF, DOCX, MD, CSV, TXT.
"""

from langchain_community.document_loaders import YoutubeLoader
from app.core.config import settings
from aiohttp.web import HTTPException
from app.core.exceptions import InternalServerError
from pathlib import Path
from typing import List
import asyncio

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from app.core.logger import logger
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.md', '.markdown', '.csv', '.txt'}


def is_supported(file_name: str) -> bool:
    """Check if the file extension is supported."""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS

async def load_web(urls: list[str]) -> list[Document]:
    """
    Concurrently fetches HTML from a list of URLs and transforms it into 
    clean, markdown-formatted LangChain Documents.
    """
    if not urls:
        return []

    try:
        loader = AsyncHtmlLoader(
            web_path=urls,
            ignore_load_errors=True # Prevents 1 broken URL from crashing the whole batch
        )
        
        # 2. Use .aload() instead of .load() to avoid blocking the FastAPI event loop
        logger.info(f"Starting concurrent download of {len(urls)} web sources...")
        raw_html_docs = await loader.aload()
        
        if not raw_html_docs:
            logger.warning("No HTML content could be successfully retrieved from the provided URLs.")
            return []

        # This strips out scripts, style blocks, and non-content boilerplate
        transformer = Html2TextTransformer()
        
        # Note: transform_documents is currently CPU-bound/sync in LangChain.
        # For huge batches of documents, running this in a separate thread prevents loop lag.
        logger.info("Transforming raw HTML payloads into structured Markdown...")
        cleaned_markdown_docs = transformer.transform_documents(raw_html_docs)
        
        return cleaned_markdown_docs

    except Exception as e:
        logger.error(f"Critical failure in web ingestion pipeline: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to process external web targets: {str(e)}")

async def load_topic(topic: str) -> list[Document]:
    """ take topi, search on interner and get websites urls using travily api 
    then get urls content using loadweb"""

    clean_topic = topic.strip()
    if not clean_topic:
        logger.warning("Received an empty topic string for ingestion.")
        return []

    try:
        logger.info(f"Initializing web search for topic: '{clean_topic}'")
        
        search_api = TavilySearchAPIWrapper(tavily_api_key=settings.TAVILY_API_KEY)
        
        raw_results = await asyncio.to_thread(
            search_api.results, 
            query=clean_topic, 
            max_results=settings.TAVILY_MAX_RESULTS
        )
        
        urls = [res["url"] for res in raw_results if isinstance(res, dict) and "url" in res]
        
        if not urls:
            logger.warning(f"Tavily returned zero valid target URLs for topic: '{clean_topic}'")
            return []
            
        logger.info(f"Discovered {len(urls)} target URLs. Handing off to the web loader layer...")

        # 4. Await your asynchronous web content loader
        # This calls the load_web function we built previously using AsyncHtmlLoader
        documents = await load_web(urls)
        
        return documents

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed search and ingestion workflow for topic '{clean_topic}': {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to orchestrate topic research pipeline: {str(e)}")

async def load_yt(url: str) -> List[Document]:
    """
    Asynchronously fetches the transcript of a YouTube video and returns it as 
    LangChain Documents.
    """
    clean_url = url.strip()
    if not clean_url:
        logger.warning("Received an empty URL string for YouTube ingestion.")
        return []

    try:
        logger.info(f"Initializing YouTube loader for URL: '{clean_url}'")
        
        loader = YoutubeLoader.from_youtube_url(
            clean_url,
            add_video_info=False,
            language=["en", "en-US","hi", "es", "fr", "de"]
        )
        
        documents = await asyncio.to_thread(loader.load)
            
        if not documents:
            logger.warning(f"No transcript could be retrieved from the video: {clean_url}.")
            return []

        # FIX: Override LangChain's default metadata to guarantee it matches your exact input URL
        for doc in documents:
            doc.metadata["source"] = clean_url
            
        return documents

    except Exception as e:
        # Log individual failures but allow the bulk orchestrator to catch/handle
        logger.error(f"Failed to load single YouTube transcript for {clean_url}: {str(e)}")
        return []

async def load_yt_bulk(urls: List[str]) -> List[Document]:
    """
    Concurrently fetches transcripts for a collection of YouTube URLs.
    Gracefully handles partial failures so one broken link won't halt the pipeline.
    """
    # Clean and filter out empty inputs
    valid_urls = [url.strip() for url in urls if url and url.strip()]
    
    if not valid_urls:
        logger.warning("Received an empty or invalid list of YouTube URLs for bulk ingestion.")
        return []

    try:
        logger.info(f"Starting concurrent ingestion of {len(valid_urls)} YouTube targets...")
        
        # Create concurrent tasks for all URLs
        tasks = [load_yt(url) for url in valid_urls]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_documents = []
        for url, result in zip(valid_urls, results):
            if isinstance(result, Exception):
                # Uncaught exception safety net for a single task
                logger.error(f"Task tracking failure for URL {url}: {str(result)}")
                continue
                
            if result:
                combined_documents.extend(result)

        logger.info(
            f"Bulk YouTube processing complete. Compiled {len(combined_documents)} total transcript documents "
            f"across requested batch."
        )
        return combined_documents

    except Exception as e:
        logger.error(f"Critical failure in bulk YouTube ingestion orchestrator: {str(e)}", exc_info=True)
        raise InternalServerError(f"Failed to orchestrate bulk YouTube research pipeline: {str(e)}")

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
