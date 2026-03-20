"""
Chunk construction for the RAG LLM Semantic Search API ingestion path.

What this module does:
    Splits page text into retrieval-sized chunks and converts each chunk into
    the exact nine-field ChunkMetadata schema used by embeddings and FAISS.

Chunking strategy:
    The splitter tries paragraph, line, word, and character boundaries in that
    order. It keeps chunks near the configured token target and carries token
    overlap forward so adjacent chunks preserve local context.

Why this module exists:
    Data quality starts here. The metadata created in this module flows through
    embeddings, vector storage, retrieval, and finally the user-facing source
    attribution returned by the API.

Execution model:
    All functions in this module are synchronous by design. Chunking is pure
    CPU work with no file or network I/O.
"""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib

import tiktoken

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ChunkMetadata


_TIKTOKEN_MODEL: str = "gpt-4o-mini"
_SPLIT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")
_HASH_ENCODING: str = "utf-8"

logger = get_logger(__name__)


def _split_by_characters(
    text: str,
    chunk_size: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Split text by character boundaries while respecting the token limit."""
    character_chunks: list[str] = []
    current_chunk = ""

    for character in text:
        candidate = f"{current_chunk}{character}"
        if current_chunk and count_tokens(candidate, encoder) > chunk_size:
            character_chunks.append(current_chunk)
            current_chunk = character
            continue
        current_chunk = candidate

    if current_chunk:
        character_chunks.append(current_chunk)

    return character_chunks


def _split_base_chunks(
    text: str,
    chunk_size: int,
    encoder: tiktoken.Encoding,
    separators: tuple[str, ...],
) -> list[str]:
    """Recursively split text into contiguous base chunks before overlap."""
    if not text:
        return []

    if count_tokens(text, encoder) <= chunk_size:
        return [text]

    if not separators:
        return _split_by_characters(text, chunk_size, encoder)

    separator = separators[0]
    if separator == "":
        return _split_by_characters(text, chunk_size, encoder)

    split_parts = text.split(separator)
    if len(split_parts) == 1:
        return _split_base_chunks(text, chunk_size, encoder, separators[1:])

    pieces = [split_parts[0]]
    pieces.extend(f"{separator}{part}" for part in split_parts[1:])

    base_chunks: list[str] = []
    current_chunk = ""
    for piece in pieces:
        if count_tokens(piece, encoder) > chunk_size:
            if current_chunk:
                base_chunks.append(current_chunk)
                current_chunk = ""
            base_chunks.extend(
                _split_base_chunks(piece, chunk_size, encoder, separators[1:])
            )
            continue

        candidate = f"{current_chunk}{piece}" if current_chunk else piece
        if current_chunk and count_tokens(candidate, encoder) > chunk_size:
            base_chunks.append(current_chunk)
            current_chunk = piece
            continue
        current_chunk = candidate

    if current_chunk:
        base_chunks.append(current_chunk)

    return base_chunks


def _get_overlap_text(
    chunk_text: str,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> str:
    """Return the trailing overlap token text for a chunk."""
    if chunk_overlap <= 0 or not chunk_text:
        return ""

    tokens = encoder.encode(chunk_text)
    if not tokens:
        return ""

    return encoder.decode(tokens[-chunk_overlap:])


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """
    Count tokens in text using the provided encoder.

    Args:
        text: Input text to measure.
        encoder: tiktoken encoder used for measurement.

    Returns:
        The token count for the input text. Empty input returns zero.
    """
    if not text:
        return 0
    return len(encoder.encode(text))


def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
    separators: tuple[str, ...] = _SPLIT_SEPARATORS,
) -> list[str]:
    """
    Split text recursively and apply token overlap between adjacent chunks.

    Args:
        text: Source text to split.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens between adjacent chunks.
        encoder: tiktoken encoder used for token counting.
        separators: Ordered separator fallback chain.

    Returns:
        A list of chunk strings. Empty input returns an empty list.
    """
    if not text.strip():
        logger.debug("Received empty text for recursive splitting.")
        return []

    base_chunks = _split_base_chunks(text, chunk_size, encoder, separators)
    if not base_chunks:
        return []

    if chunk_overlap <= 0:
        return base_chunks

    overlapped_chunks = [base_chunks[0]]
    for index in range(1, len(base_chunks)):
        overlap_text = _get_overlap_text(
            base_chunks[index - 1],
            chunk_overlap,
            encoder,
        )
        current_chunk = base_chunks[index]
        if overlap_text:
            current_chunk = f"{overlap_text}{current_chunk}"
        overlapped_chunks.append(current_chunk)

    return overlapped_chunks


def build_chunks(
    pages: list[tuple[int | None, str]],
    source_file: str,
    namespace: str,
) -> list[ChunkMetadata]:
    """
    Convert loaded page text into ChunkMetadata objects.

    Args:
        pages: Page content tuples from the document loader.
        source_file: Original source filename.
        namespace: Namespace assigned to the document ingestion.

    Returns:
        A list of ChunkMetadata objects ready for embedding and storage.
    """
    if not pages:
        logger.debug(
            "No pages were provided for chunk construction.",
            extra={"source_file": source_file},
        )
        return []

    settings = get_settings()
    encoder = get_encoder()
    metadata_chunks: list[ChunkMetadata] = []
    chunk_index = 0

    for page_number, page_text in pages:
        text_chunks = split_text_recursive(
            text=page_text,
            chunk_size=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
            encoder=encoder,
        )
        if not text_chunks:
            logger.warning(
                "Page produced zero chunks.",
                extra={
                    "source_file": source_file,
                    "page_number": page_number,
                },
            )
            continue

        for chunk_text in text_chunks:
            stripped_chunk = chunk_text.strip()
            if not stripped_chunk:
                logger.debug(
                    "Skipping empty chunk after splitting.",
                    extra={
                        "source_file": source_file,
                        "page_number": page_number,
                    },
                )
                continue

            metadata_chunks.append(
                ChunkMetadata(
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    char_start=page_text.find(chunk_text),
                    token_count=count_tokens(chunk_text, encoder),
                    content_hash=hashlib.sha256(
                        stripped_chunk.encode(_HASH_ENCODING)
                    ).hexdigest(),
                    ingested_at=datetime.now(timezone.utc),
                    namespace=namespace,
                )
            )
            chunk_index += 1

    if not metadata_chunks:
        logger.warning(
            "Document produced zero chunks.",
            extra={"source_file": source_file},
        )
        return []

    logger.debug(
        "Built document chunks.",
        extra={
            "source_file": source_file,
            "chunk_count": len(metadata_chunks),
        },
    )
    return metadata_chunks


@lru_cache(maxsize=1)
def get_encoder() -> tiktoken.Encoding:
    """
    Return the cached tiktoken encoder used for chunk measurement.

    Returns:
        The encoder for the configured completion-compatible tokenisation
        model.
    """
    return tiktoken.encoding_for_model(_TIKTOKEN_MODEL)
