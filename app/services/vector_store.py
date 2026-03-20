"""
FAISS storage layer for the RAG LLM Semantic Search API.

Responsibilities:
    Own every direct FAISS interaction, including namespace preload,
    in-memory index management, on-disk persistence, and similarity
    search against the loaded namespaces.

Why this module exists:
    The vector store sits underneath ingestion and retrieval. It must
    prevent concurrent write corruption, keep CPU-bound work off the
    event loop, and make namespace-level failure modes explicit.

Storage contract:
    Each namespace persists as two companion files in the configured
    index directory: one `.index` file for the FAISS binary payload and
    one `.metadata` JSON file containing ChunkMetadata dictionaries.

Concurrency model:
    Each namespace has a dedicated asyncio.Lock. Every CPU-bound FAISS,
    numpy, and serialisation operation runs through a shared
    ThreadPoolExecutor owned by the service instance.
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
from typing import TypedDict

import faiss  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ChunkMetadata


_INDEX_SUFFIX: str = ".index"
_METADATA_SUFFIX: str = ".metadata"
_FAISS_DIMENSION: int = 1536

logger = get_logger(__name__)


class ChunkMetadataRecord(TypedDict):
    """JSON-serialisable ChunkMetadata payload stored on disk."""

    text: str
    source_file: str
    page_number: int | None
    chunk_index: int
    char_start: int
    token_count: int
    content_hash: str
    ingested_at: str
    namespace: str


def _build_metadata_record(chunk: ChunkMetadata) -> ChunkMetadataRecord:
    """Convert a ChunkMetadata model into a JSON-safe metadata record."""
    return {
        "text": chunk.text,
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "token_count": chunk.token_count,
        "content_hash": chunk.content_hash,
        "ingested_at": chunk.ingested_at.isoformat(),
        "namespace": chunk.namespace,
    }


def _build_chunk_metadata(record: ChunkMetadataRecord) -> ChunkMetadata:
    """Convert a JSON metadata record back into a ChunkMetadata model."""
    return ChunkMetadata(
        text=record["text"],
        source_file=record["source_file"],
        page_number=record["page_number"],
        chunk_index=record["chunk_index"],
        char_start=record["char_start"],
        token_count=record["token_count"],
        content_hash=record["content_hash"],
        ingested_at=record["ingested_at"],
        namespace=record["namespace"],
    )


def _list_index_paths(index_dir: Path) -> list[Path]:
    """Return all index file paths in the configured index directory."""
    return list(index_dir.glob(f"*{_INDEX_SUFFIX}"))


def _load_metadata_records(metadata_path: Path) -> list[ChunkMetadataRecord]:
    """Load and validate chunk metadata records from disk."""
    raw_value: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, list):
        raise ValueError("metadata file must contain a JSON list")

    records: list[ChunkMetadataRecord] = []
    for item in raw_value:
        if not isinstance(item, dict):
            raise ValueError("metadata entries must be JSON objects")

        text = item.get("text")
        source_file = item.get("source_file")
        page_number = item.get("page_number")
        chunk_index = item.get("chunk_index")
        char_start = item.get("char_start")
        token_count = item.get("token_count")
        content_hash = item.get("content_hash")
        ingested_at = item.get("ingested_at")
        namespace = item.get("namespace")

        if not isinstance(text, str):
            raise ValueError("metadata text must be a string")
        if not isinstance(source_file, str):
            raise ValueError("metadata source_file must be a string")
        if page_number is not None and not isinstance(page_number, int):
            raise ValueError("metadata page_number must be an int or null")
        if not isinstance(chunk_index, int):
            raise ValueError("metadata chunk_index must be an int")
        if not isinstance(char_start, int):
            raise ValueError("metadata char_start must be an int")
        if not isinstance(token_count, int):
            raise ValueError("metadata token_count must be an int")
        if not isinstance(content_hash, str):
            raise ValueError("metadata content_hash must be a string")
        if not isinstance(ingested_at, str):
            raise ValueError("metadata ingested_at must be a string")
        if not isinstance(namespace, str):
            raise ValueError("metadata namespace must be a string")

        records.append(
            {
                "text": text,
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "char_start": char_start,
                "token_count": token_count,
                "content_hash": content_hash,
                "ingested_at": ingested_at,
                "namespace": namespace,
            }
        )
    return records


def _to_vectors_array(vectors: list[list[float]]) -> npt.NDArray[np.float32]:
    """Convert a vector batch into a 2D float32 numpy array."""
    vectors_array = np.asarray(vectors, dtype=np.float32)
    if vectors_array.ndim != 2:
        raise ValueError("vectors must form a 2D array")
    if vectors_array.shape[1] != _FAISS_DIMENSION:
        raise ValueError(
            f"vectors must have dimension {_FAISS_DIMENSION}. "
            f"Got: {vectors_array.shape[1]}"
        )
    return vectors_array


def _to_query_array(query_vector: list[float]) -> npt.NDArray[np.float32]:
    """Convert a single query vector into a 2D float32 numpy array."""
    query_array = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if query_array.shape[1] != _FAISS_DIMENSION:
        raise ValueError(
            f"query vector must have dimension {_FAISS_DIMENSION}. "
            f"Got: {query_array.shape[1]}"
        )
    return query_array


class VectorStoreService:
    """Namespaced FAISS storage service with async-safe mutation semantics."""

    def __init__(self) -> None:
        """Initialise in-memory indexes, metadata, locks, and executor."""
        settings = get_settings()
        self._indexes: dict[str, faiss.IndexFlatIP] = {}
        self._metadata: dict[str, list[ChunkMetadataRecord]] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._executor = ThreadPoolExecutor()
        self._index_dir = Path(settings.FAISS_INDEX_DIR)

    async def preload_all_namespaces(self) -> None:
        """
        Load every persisted namespace into memory during application startup.

        Returns:
            None.
        """
        loop = asyncio.get_running_loop()

        if not await loop.run_in_executor(self._executor, self._index_dir.exists):
            await loop.run_in_executor(
                self._executor,
                partial(self._index_dir.mkdir, parents=True, exist_ok=True),
            )
            logger.info(
                "Created FAISS index directory because it did not exist.",
                extra={"index_dir": str(self._index_dir)},
            )
            logger.info("Completed FAISS namespace preload.", extra={"count": 0})
            return

        index_paths = await loop.run_in_executor(
            self._executor,
            partial(_list_index_paths, self._index_dir),
        )

        loaded_count = 0
        for index_path in index_paths:
            namespace = index_path.stem
            metadata_path = index_path.with_suffix(_METADATA_SUFFIX)
            metadata_exists = await loop.run_in_executor(
                self._executor,
                metadata_path.is_file,
            )
            if not metadata_exists:
                logger.warning(
                    "Skipping namespace preload because metadata is missing.",
                    extra={"namespace": namespace},
                )
                continue

            try:
                index, metadata = await loop.run_in_executor(
                    self._executor,
                    self._sync_load_index,
                    index_path,
                    metadata_path,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                logger.error(
                    "Failed to preload namespace from disk.",
                    exc_info=True,
                    extra={"namespace": namespace},
                )
                continue

            self._indexes[namespace] = index
            self._metadata[namespace] = metadata
            loaded_count += 1

        logger.info(
            "Completed FAISS namespace preload.",
            extra={"count": loaded_count},
        )

    async def add_vectors(
        self,
        namespace: str,
        vectors: list[list[float]],
        metadata: list[ChunkMetadata],
    ) -> None:
        """
        Add vectors and metadata to a namespace, then persist both to disk.

        Returns:
            None.

        Raises:
            ValueError: If vectors and metadata counts do not match or if the
                vector dimension is not compatible with the configured index.
        """
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        if not vectors:
            logger.debug(
                "No vectors were provided for add operation.",
                extra={"namespace": namespace, "vector_count": 0},
            )
            return

        loop = asyncio.get_running_loop()
        vectors_array = await loop.run_in_executor(
            self._executor,
            partial(_to_vectors_array, vectors),
        )
        metadata_records = [_build_metadata_record(item) for item in metadata]

        async with self._locks[namespace]:
            if namespace not in self._indexes:
                self._indexes[namespace] = faiss.IndexFlatIP(_FAISS_DIMENSION)
                self._metadata[namespace] = []

            await loop.run_in_executor(
                self._executor,
                self._sync_add,
                self._indexes[namespace],
                vectors_array,
            )
            self._metadata[namespace].extend(metadata_records)
            await loop.run_in_executor(
                self._executor,
                self._sync_save_index,
                namespace,
            )

        logger.debug(
            "Added vectors to namespace.",
            extra={"namespace": namespace, "vector_count": len(vectors)},
        )

    async def search(
        self,
        namespace: str,
        query_vector: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[tuple[ChunkMetadata, float]]:
        """
        Search a namespace and return threshold-filtered chunk matches.

        Returns:
            A descending list of (ChunkMetadata, score) tuples.
        """
        if namespace not in self._indexes:
            logger.warning(
                "Requested namespace is not loaded in memory.",
                extra={"namespace": namespace},
            )
            return []

        loop = asyncio.get_running_loop()
        query_array = await loop.run_in_executor(
            self._executor,
            partial(_to_query_array, query_vector),
        )
        scores, indices = await loop.run_in_executor(
            self._executor,
            self._sync_search,
            namespace,
            query_array,
            top_k,
        )

        namespace_metadata = self._metadata.get(namespace, [])
        results: list[tuple[ChunkMetadata, float]] = []
        for raw_score, raw_index in zip(scores[0], indices[0]):
            index_value = int(raw_index)
            if index_value < 0:
                continue
            if index_value >= len(namespace_metadata):
                logger.warning(
                    "FAISS returned an out-of-range metadata index.",
                    extra={"namespace": namespace, "index": index_value},
                )
                continue

            score = float(raw_score)
            if score < similarity_threshold:
                continue

            chunk = _build_chunk_metadata(namespace_metadata[index_value])
            results.append((chunk, score))

        results.sort(key=lambda item: item[1], reverse=True)
        logger.debug(
            "Completed FAISS search.",
            extra={
                "namespace": namespace,
                "top_k": top_k,
                "result_count": len(results),
            },
        )
        return results

    async def get_namespaces(self) -> list[str]:
        """
        Return the loaded namespace names in sorted order.

        Returns:
            Sorted namespace names currently resident in memory.
        """
        return sorted(self._indexes.keys())

    async def namespace_exists(self, namespace: str) -> bool:
        """
        Check whether a namespace is loaded in memory.

        Returns:
            True when the namespace is present, otherwise False.
        """
        return namespace in self._indexes

    def _sync_search(
        self,
        namespace: str,
        query_array: npt.NDArray[np.float32],
        top_k: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Run a FAISS search synchronously inside the shared executor."""
        return self._indexes[namespace].search(query_array, top_k)

    def _sync_add(
        self,
        index: faiss.IndexFlatIP,
        vectors_array: npt.NDArray[np.float32],
    ) -> None:
        """Add a vector batch to a FAISS index inside the shared executor."""
        index.add(vectors_array)

    def _sync_save_index(self, namespace: str) -> None:
        """Persist the FAISS index and companion metadata file to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_dir / f"{namespace}{_INDEX_SUFFIX}"
        metadata_path = self._index_dir / f"{namespace}{_METADATA_SUFFIX}"
        faiss.write_index(self._indexes[namespace], str(index_path))
        metadata_json = json.dumps(self._metadata[namespace])
        metadata_path.write_text(metadata_json, encoding="utf-8")

    def _sync_load_index(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> tuple[faiss.IndexFlatIP, list[ChunkMetadataRecord]]:
        """Load a FAISS index and companion metadata file from disk."""
        index = faiss.read_index(str(index_path))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("loaded index must be an IndexFlatIP instance")

        metadata = _load_metadata_records(metadata_path)
        return index, metadata


def create_vector_store_service() -> VectorStoreService:
    """
    Create a VectorStoreService without preloading namespaces.

    Returns:
        A configured VectorStoreService instance.
    """
    return VectorStoreService()
