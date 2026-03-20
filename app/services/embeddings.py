"""
Embedding service for the RAG LLM Semantic Search API.

What this module does:
    Provides the single embedding boundary used by ingestion and retrieval.
    It embeds document chunks for ingestion and embeds one query string for
    retrieval.

Why dedup matters:
    Chunk embeddings are the most expensive step in the ingestion pipeline.
    Re-embedding content that was already processed wastes OpenAI spend and
    creates avoidable pressure on throughput and rate limits.
    Dedup is also a correctness guard. A duplicate chunk should be skipped
    before the API call, not embedded again and written twice downstream.

Async contract:
    This module uses openai.AsyncOpenAI exclusively.
    Every network call is awaited.
    The client is created once and reused for the lifetime of the service.
    No synchronous OpenAI client is imported or used anywhere here.

Error handling contract:
    OpenAI API failures are logged at ERROR with exc_info=True and re-raised.
    The embedding layer never swallows an API failure and never returns empty
    vectors as a fallback.

Two distinct operations:
    Ingest embeddings deduplicate by content hash before batching.
    Query embedding performs one direct embedding call with no dedup and no
    caching at this layer.
"""

import time

import openai

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ChunkMetadata
from app.services.cache import CacheService


_OPENAI_BATCH_SIZE: int = 512
_EMBEDDING_INPUT_TYPE: str = "float"
_MILLISECONDS_PER_SECOND: float = 1000.0

logger = get_logger(__name__)


class EmbeddingsService:
    """Async embedding service for ingestion chunks and retrieval queries."""

    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        cache_service: CacheService,
    ) -> None:
        """
        Store the shared OpenAI client, cache service, and settings.

        Args:
            openai_client: Reusable AsyncOpenAI client for embedding calls.
            cache_service: Cache service used for document-hash deduplication.
        """
        self._client = openai_client
        self._cache_service = cache_service
        self._settings = get_settings()

    async def embed_chunks(
        self,
        chunks: list[ChunkMetadata],
    ) -> list[tuple[ChunkMetadata, list[float]]]:
        """
        Embed ingestion chunks after content-hash deduplication.

        Args:
            chunks: Chunk metadata objects to embed for ingestion.

        Returns:
            A list of ``(ChunkMetadata, vector)`` tuples for non-duplicate
            chunks that were successfully embedded.

        Raises:
            openai.RateLimitError: Propagated from the OpenAI API call.
            openai.APITimeoutError: Propagated from the OpenAI API call.
            openai.APIConnectionError: Propagated from the OpenAI API call.
            openai.APIStatusError: Propagated from the OpenAI API call.
        """
        start_time = time.perf_counter()
        skipped_count = 0
        unknown_chunks: list[ChunkMetadata] = []

        for chunk in chunks:
            if await self._cache_service.is_doc_hash_known(chunk.content_hash):
                skipped_count += 1
                logger.warning(
                    "Skipping duplicate chunk before embedding.",
                    extra={
                        "source_file": chunk.source_file,
                        "content_hash": chunk.content_hash,
                    },
                )
                continue
            unknown_chunks.append(chunk)

        embedded_chunks: list[tuple[ChunkMetadata, list[float]]] = []
        for start_index in range(0, len(unknown_chunks), _OPENAI_BATCH_SIZE):
            batch_chunks = unknown_chunks[start_index : start_index + _OPENAI_BATCH_SIZE]
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_vectors = await self._embed_batch(batch_texts)

            if len(batch_vectors) != len(batch_chunks):
                raise RuntimeError(
                    "OpenAI embedding response count did not match input count."
                )

            for chunk, vector in zip(batch_chunks, batch_vectors):
                await self._cache_service.register_doc_hash(chunk.content_hash)
                embedded_chunks.append((chunk, vector))

        latency_ms = int(
            (time.perf_counter() - start_time) * _MILLISECONDS_PER_SECOND
        )
        logger.info(
            "Completed chunk embedding.",
            extra={
                "total_count": len(chunks),
                "embedded_count": len(embedded_chunks),
                "skipped_count": skipped_count,
                "latency_ms": latency_ms,
            },
        )
        return embedded_chunks

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Args:
            query: Query text to embed.

        Returns:
            The embedding vector for the query as a list of floats.

        Raises:
            openai.RateLimitError: Propagated from the OpenAI API call.
            openai.APITimeoutError: Propagated from the OpenAI API call.
            openai.APIConnectionError: Propagated from the OpenAI API call.
            openai.APIStatusError: Propagated from the OpenAI API call.
        """
        start_time = time.perf_counter()
        query_vectors = await self._embed_batch([query])
        if len(query_vectors) != 1:
            raise RuntimeError(
                "OpenAI query embedding response count did not equal one."
            )

        latency_ms = int(
            (time.perf_counter() - start_time) * _MILLISECONDS_PER_SECOND
        )
        logger.debug(
            "Embedded retrieval query.",
            extra={
                "query_length": len(query),
                "latency_ms": latency_ms,
            },
        )
        return query_vectors[0]

    async def _embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Call the OpenAI embeddings API for a batch of texts.

        Args:
            texts: Input strings to embed in one API request.

        Returns:
            Embedding vectors in the same order as the input list.

        Raises:
            openai.RateLimitError: If the OpenAI rate limit is reached.
            openai.APITimeoutError: If the OpenAI request times out.
            openai.APIConnectionError: If the OpenAI request cannot connect.
            openai.APIStatusError: If the OpenAI API returns a non-2xx status.
        """
        if not texts:
            return []

        start_time = time.perf_counter()
        try:
            response = await self._client.embeddings.create(
                model=self._settings.OPENAI_EMBEDDING_MODEL,
                input=texts,
                encoding_format=_EMBEDDING_INPUT_TYPE,
            )
        except openai.RateLimitError:
            logger.error("OpenAI rate limit reached", exc_info=True)
            raise
        except openai.APITimeoutError:
            logger.error("OpenAI API timeout", exc_info=True)
            raise
        except openai.APIConnectionError:
            logger.error("OpenAI API connection error", exc_info=True)
            raise
        except openai.APIStatusError as exc:
            logger.error(
                "OpenAI API status error",
                exc_info=True,
                extra={
                    "status_code": exc.status_code,
                    "message": str(exc),
                },
            )
            raise

        sorted_data = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in sorted_data]

        latency_ms = int(
            (time.perf_counter() - start_time) * _MILLISECONDS_PER_SECOND
        )
        logger.debug(
            "Embedded batch via OpenAI.",
            extra={
                "batch_size": len(texts),
                "latency_ms": latency_ms,
            },
        )
        return vectors


def create_embeddings_service(
    cache_service: CacheService,
) -> EmbeddingsService:
    """
    Create an EmbeddingsService with a single reusable AsyncOpenAI client.

    Args:
        cache_service: Cache service used for document-hash deduplication.

    Returns:
        A configured EmbeddingsService instance.
    """
    openai_client = openai.AsyncOpenAI()
    return EmbeddingsService(
        openai_client=openai_client,
        cache_service=cache_service,
    )
