"""
API schema contracts for the RAG LLM Semantic Search API.

What this module contains:
    Every request and response model shared by routes and services.
    All models use Pydantic v2 and freeze instances after validation.

Why this module exists:
    These schemas define the public API contract and the internal
    structured payloads that move through ingestion, retrieval,
    streaming, admin key management, and health reporting paths.

How to use it:
    Route handlers import request and response models from here.
    Services use the shared metadata and SSE payload models directly.

Validation guarantees:
    UUIDs are typed as uuid.UUID, timestamps are typed as datetime,
    and boundary checks fail with human-readable ValueError messages.

Change policy:
    Modify this file only when the API contract changes in the
    architecture reference document.
"""

from datetime import datetime
import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MIN_QUERY_LENGTH: int = 1
_MAX_QUERY_LENGTH: int = 2000
_MIN_TOP_K: int = 1
_DEFAULT_NAMESPACE: str = "default"
_JOB_STATUSES: frozenset[str] = frozenset(
    {"queued", "processing", "done", "failed"}
)


class ChunkSource(BaseModel):
    """Source attribution for a retrieved chunk in QueryResponse.sources."""

    model_config = ConfigDict(frozen=True)

    # Original filename the chunk was retrieved from.
    file: str
    # 1-indexed page number. None for .txt source files.
    page: int | None
    # 0-indexed position of this chunk in its source file.
    chunk_index: int
    # Cosine similarity score. Must be between 0.0 and 1.0 inclusive.
    score: Annotated[float, Field()]

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"score must be between 0.0 and 1.0 inclusive. Got: {value}"
            )
        return value


class ChunkMetadata(BaseModel):
    """Full 9-field FAISS chunk metadata used by ingest and retrieval code."""

    model_config = ConfigDict(frozen=True)

    # Raw text content of this chunk.
    text: str
    # Original filename e.g. report.pdf.
    source_file: str
    # 1-indexed page number. None for .txt files.
    page_number: int | None
    # 0-indexed position within the source file.
    chunk_index: int
    # Character offset of chunk start in the source document.
    char_start: int
    # Actual token count via tiktoken.
    token_count: int
    # SHA256 of normalised chunk text for dedup and embedding cache keys.
    content_hash: str
    # UTC timestamp of ingestion.
    ingested_at: datetime
    # Namespace this chunk belongs to.
    namespace: str


class IngestResponse(BaseModel):
    """Response returned by POST /ingest after queuing a job."""

    model_config = ConfigDict(frozen=True)

    # Unique identifier for this ingestion job.
    job_id: uuid.UUID
    # Initial status for an accepted ingestion request.
    status: str
    # Number of files accepted for ingestion.
    file_count: int
    # Target namespace for this ingestion job.
    namespace: str


class JobStatusResponse(BaseModel):
    """Response returned by GET /ingest/{job_id} for job progress polling."""

    model_config = ConfigDict(frozen=True)

    # Unique identifier for this ingestion job.
    job_id: uuid.UUID
    # Current job status. Must be queued, processing, done, or failed.
    status: str
    # Number of chunks successfully indexed so far.
    chunks_indexed: int
    # Number of files fully processed so far.
    files_processed: int
    # Error messages accumulated during processing.
    errors: list[str]
    # UTC completion timestamp. None while the job is still running.
    completed_at: datetime | None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in _JOB_STATUSES:
            valid_statuses = ", ".join(sorted(_JOB_STATUSES))
            raise ValueError(
                f"status must be one of: {valid_statuses}. Got: {value!r}"
            )
        return value


class QueryRequest(BaseModel):
    """Request body for POST /query consumed by the RAG pipeline."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # The user question. Required length: 1 to 2000 characters.
    query: Annotated[
        str,
        Field(min_length=_MIN_QUERY_LENGTH, max_length=_MAX_QUERY_LENGTH),
    ]
    # Target FAISS namespace. Default: "default".
    namespace: str = _DEFAULT_NAMESPACE
    # Existing session UUID. None creates a new session.
    session_id: uuid.UUID | None = None
    # Chunks to retrieve. Must be between 1 and settings.TOP_K_MAX.
    top_k: Annotated[int, Field()] = 5
    # Request-level rerank override. None means use the global setting.
    rerank: bool | None = None
    # Enable SSE streaming. False returns the standard JSON response.
    stream: bool = True

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        from app.core.config import get_settings

        settings = get_settings()
        if not _MIN_TOP_K <= value <= settings.TOP_K_MAX:
            raise ValueError(
                "top_k must be between "
                f"{_MIN_TOP_K} and {settings.TOP_K_MAX}. Got: {value}"
            )
        return value


class QueryResponse(BaseModel):
    """Response returned by POST /query for JSON and cached code paths."""

    model_config = ConfigDict(frozen=True)

    # Generated answer or the insufficient-context fallback message.
    answer: str
    # Session UUID for the current conversation.
    session_id: uuid.UUID
    # Retrieved chunk sources used to build the answer.
    sources: list[ChunkSource]
    # True when Redis served this response from cache.
    cached: bool
    # True when cross-encoder re-ranking was applied.
    reranked: bool
    # True when the rewrite heuristic produced a rewritten query.
    query_rewritten: bool
    # Query text exactly as submitted by the caller.
    original_query: str
    # Rewritten query text. None when query_rewritten is False.
    rewritten_query: str | None
    # Total request latency in milliseconds.
    latency_ms: int


class SSEEvent(BaseModel):
    """Single SSE frame used by routes/query.py to build the byte stream."""

    model_config = ConfigDict(frozen=True)

    # Event type. Must be metadata, chunk, sources, or done.
    event: str
    # JSON-serialised payload for this event.
    data: str

    @field_validator("event")
    @classmethod
    def validate_event(cls, value: str) -> str:
        allowed_events = ("metadata", "chunk", "sources", "done")
        if value not in allowed_events:
            raise ValueError(
                f"event must be one of: {', '.join(allowed_events)}. "
                f"Got: {value!r}"
            )
        return value


class SSEMetadata(BaseModel):
    """Metadata payload emitted as the first SSE event for a streamed query."""

    model_config = ConfigDict(frozen=True)

    # Session UUID available to the client before token streaming begins.
    session_id: uuid.UUID
    # True when the response came from cache.
    cached: bool
    # True when cross-encoder re-ranking was applied.
    reranked: bool
    # True when the query rewrite heuristic fired.
    query_rewritten: bool
    # Rewritten query text. None when no rewrite occurred.
    rewritten_query: str | None


class SSEChunk(BaseModel):
    """Token payload emitted for each chunk SSE event."""

    model_config = ConfigDict(frozen=True)

    # A single streamed token from the LLM completion.
    token: str


class SSEDone(BaseModel):
    """Final SSE payload emitted after streaming completes."""

    model_config = ConfigDict(frozen=True)

    # Total request latency in milliseconds.
    latency_ms: int


class DeleteSessionResponse(BaseModel):
    """Response returned by DELETE /session/{session_id} on success."""

    model_config = ConfigDict(frozen=True)

    # Always True when the session delete operation succeeds.
    deleted: bool
    # Session UUID that was deleted.
    session_id: uuid.UUID


class AdminKeyRequest(BaseModel):
    """Request body for POST /admin/keys when provisioning a new key."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Human-readable label for this API key. Length: 1 to 100 characters.
    name: Annotated[str, Field(min_length=1, max_length=100)]


class AdminKeyResponse(BaseModel):
    """Response returned by POST /admin/keys with the one-time plaintext key."""

    model_config = ConfigDict(frozen=True)

    # Unique identifier for this key record.
    key_id: uuid.UUID
    # Plaintext API key in rag_<64-char-hex> format.
    key: str
    # Human-readable label as submitted in the request.
    name: str
    # UTC timestamp of key creation.
    created_at: datetime


class AdminKeyMetadata(BaseModel):
    """Metadata record returned by GET /admin/keys without the plaintext key."""

    model_config = ConfigDict(frozen=True)

    # Unique identifier for this key record.
    key_id: uuid.UUID
    # Human-readable label for this key record.
    name: str
    # UTC timestamp when this key was created.
    created_at: datetime
    # UTC timestamp of last successful use. None if never used.
    last_used_at: datetime | None


class AdminKeyListResponse(BaseModel):
    """Response body returned by GET /admin/keys."""

    model_config = ConfigDict(frozen=True)

    # All provisioned key records. Empty list if none exist.
    keys: list[AdminKeyMetadata]


class RevokeKeyResponse(BaseModel):
    """Response returned by DELETE /admin/keys/{key_id} on success."""

    model_config = ConfigDict(frozen=True)

    # Always True when the revoke operation succeeds.
    revoked: bool
    # Unique identifier for the revoked key record.
    key_id: uuid.UUID


class HealthResponse(BaseModel):
    """Health response returned by GET /health for service readiness checks."""

    model_config = ConfigDict(frozen=True)

    # Overall service status. Expected values are "ok" or "degraded".
    status: str
    # Redis connectivity status. Expected values are "ok" or "unavailable".
    redis: str
    # Number of FAISS namespaces currently loaded in memory.
    faiss_namespaces_loaded: int
    # True when the cross-encoder reranker is loaded and ready.
    reranker_loaded: bool
    # Application version string sourced from settings.
    version: str


__all__ = [
    "ChunkSource",
    "ChunkMetadata",
    "IngestResponse",
    "JobStatusResponse",
    "QueryRequest",
    "QueryResponse",
    "SSEEvent",
    "SSEMetadata",
    "SSEChunk",
    "SSEDone",
    "DeleteSessionResponse",
    "AdminKeyRequest",
    "AdminKeyResponse",
    "AdminKeyMetadata",
    "AdminKeyListResponse",
    "RevokeKeyResponse",
    "HealthResponse",
]
