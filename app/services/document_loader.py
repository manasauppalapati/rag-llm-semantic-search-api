"""
Document loading for the RAG LLM Semantic Search API ingestion path.

What this module does:
    Loads supported document types from disk and normalises them into a
    consistent list of page-oriented text tuples for downstream chunking.

Supported formats:
    PDF files are parsed with pymupdf and returned as one tuple per page.
    Plain-text files are read as UTF-8 and returned as a single tuple with
    page_number set to None.

Why this module exists:
    The chunker and embedding pipeline need one stable input contract. This
    module owns format detection, parser dispatch, text extraction, and the
    fail-fast behavior for unreadable or textless documents.

Async discipline:
    All file I/O and PDF parsing run in a shared ThreadPoolExecutor through
    asyncio.get_running_loop().run_in_executor(). No blocking file access is
    performed on the event loop thread.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz

from app.core.logging import get_logger


_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})
_PDF_EXTENSION: str = ".pdf"
_TXT_EXTENSION: str = ".txt"
_TXT_ENCODING: str = "utf-8"
_executor: ThreadPoolExecutor = ThreadPoolExecutor()
PageContent = tuple[int | None, str]

logger = get_logger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


def _read_txt_sync(file_path: Path) -> list[PageContent]:
    """Read a UTF-8 text document synchronously for executor offloading."""
    return [(None, file_path.read_text(encoding=_TXT_ENCODING))]


async def load_document(file_path: Path) -> list[PageContent]:
    """
    Load a supported document and return filtered page text tuples.

    Args:
        file_path: Filesystem path to the input document.

    Returns:
        A list of ``(page_number, text)`` tuples with empty extracted pages
        removed.

    Raises:
        DocumentLoadError: If the file extension is unsupported or if the
            document cannot yield any extractable text.
    """
    extension = file_path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported document type for '{file_path.name}': "
            f"{extension or '<no extension>'}"
        )

    if extension == _PDF_EXTENSION:
        page_content = await _load_pdf(file_path)
    else:
        page_content = await _load_txt(file_path)

    filtered_content: list[PageContent] = []
    for page_number, text in page_content:
        if not text.strip():
            logger.warning(
                "Skipping empty extracted page.",
                extra={
                    "file_name": file_path.name,
                    "page_number": page_number,
                },
            )
            continue
        filtered_content.append((page_number, text))

    if not filtered_content:
        raise DocumentLoadError(
            f"Document '{file_path.name}' contains no extractable text."
        )

    logger.debug(
        "Loaded document content.",
        extra={
            "file_name": file_path.name,
            "page_count": len(filtered_content),
        },
    )
    return filtered_content


async def _load_pdf(file_path: Path) -> list[PageContent]:
    """
    Load a PDF document and return one text tuple per page.

    Args:
        file_path: Filesystem path to the PDF document.

    Returns:
        A list of page-numbered text tuples including empty extracted pages.

    Raises:
        DocumentLoadError: If the PDF cannot be opened or parsed.
    """
    def _read_pdf() -> list[PageContent]:
        document: fitz.Document | None = None
        try:
            document = fitz.open(str(file_path))
            return [(page.number + 1, page.get_text()) for page in document]
        finally:
            if document is not None:
                document.close()

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _read_pdf)
    except Exception as exc:
        logger.error(
            "Failed to load PDF document.",
            exc_info=True,
            extra={"file_name": file_path.name},
        )
        raise DocumentLoadError(
            f"Failed to load PDF document '{file_path.name}'."
        ) from exc


async def _load_txt(file_path: Path) -> list[PageContent]:
    """
    Load a UTF-8 text document and return its content as a single tuple.

    Args:
        file_path: Filesystem path to the text document.

    Returns:
        A single-item list containing ``(None, full_text)``.

    Raises:
        DocumentLoadError: If the text file cannot be read.
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _read_txt_sync, file_path)
    except Exception as exc:
        logger.error(
            "Failed to load text document.",
            exc_info=True,
            extra={"file_name": file_path.name},
        )
        raise DocumentLoadError(
            f"Failed to load text document '{file_path.name}'."
        ) from exc
