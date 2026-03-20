"""
Application configuration for the RAG semantic search API lives in this module.
It centralizes every environment-dependent value the service needs before any
other component starts doing work.
Application code must load settings through get_settings() so all consumers see
the same validated, cached configuration object.
Required environment variables such as OPENAI_API_KEY and ADMIN_SECRET have no
defaults, so missing values raise a ValidationError during startup.
That fail-fast behavior prevents silent misconfiguration from reaching
production and surfacing as a harder-to-debug incident later.
"""

from functools import lru_cache
from typing import Self

from pydantic import ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings


ALLOWED_LOG_LEVELS: tuple[str, ...] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)
ALLOWED_APP_ENVS: tuple[str, ...] = (
    "development",
    "staging",
    "production",
)
SIMILARITY_THRESHOLD_MIN: float = 0.0
SIMILARITY_THRESHOLD_MAX: float = 1.0

__all__ = ("get_settings",)


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- OpenAI --------------------------------------------------------------
    # Required. No default. Missing = ValidationError on startup.
    # Used by: services/embeddings.py, services/rag_pipeline.py,
    # services/query_rewriter.py
    OPENAI_API_KEY: str

    # Embedding model for document chunks and query vectors.
    # text-embedding-3-small: best cost/quality ratio at 1536 dims.
    # Used by: services/embeddings.py
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Completion model for RAG answers and query rewriting.
    # gpt-4o-mini: 128k context, low cost, sufficient quality.
    # Used by: services/rag_pipeline.py, services/query_rewriter.py
    OPENAI_COMPLETION_MODEL: str = "gpt-4o-mini"

    # Max tokens for completion responses.
    # Bounds cost and latency per request.
    # Used by: services/rag_pipeline.py
    OPENAI_MAX_TOKENS: int = 1000

    # -- Redis ---------------------------------------------------------------
    # Full Redis connection URL including scheme and port.
    # In Docker Compose this is: redis://redis:6379
    # Used by: services/cache.py
    REDIS_URL: str = "redis://localhost:6379"

    # TTL in seconds for semantic answer cache entries.
    # Key pattern: cache:answer:{sha256_of_rounded_embedding}
    # Used by: services/cache.py
    REDIS_TTL_ANSWER: int = 3600

    # TTL in seconds for conversation session history.
    # Refreshed on every access. Key pattern: session:{session_id}
    # Used by: services/cache.py
    REDIS_TTL_SESSION: int = 3600

    # TTL in seconds for ingestion job state records.
    # 24h gives time to poll status after async ingestion completes.
    # Key pattern: job:{job_id}. Used by: services/cache.py
    REDIS_TTL_JOB: int = 86400

    # TTL in seconds for rate limit token bucket entries.
    # 2x the rate window to allow natural expiry for inactive clients.
    # Key pattern: rate:{key_id}. Used by: middleware/rate_limit.py
    REDIS_TTL_RATE_BUCKET: int = 120

    # -- FAISS ---------------------------------------------------------------
    # Directory where namespaced FAISS indexes are persisted.
    # In Docker this maps to the bind-mount volume.
    # Each namespace produces: {namespace}.index + {namespace}.metadata
    # Used by: services/vector_store.py
    FAISS_INDEX_DIR: str = "./faiss_indexes"

    # -- Chunking ------------------------------------------------------------
    # Target size of each document chunk in tokens.
    # 512 balances context richness with retrieval precision.
    # Used by: services/chunker.py
    CHUNK_SIZE_TOKENS: int = 512

    # Token overlap between adjacent chunks.
    # Preserves meaning across chunk boundaries.
    # Used by: services/chunker.py
    CHUNK_OVERLAP_TOKENS: int = 50

    # -- Retrieval -----------------------------------------------------------
    # Default number of chunks returned by FAISS per query.
    # Callers may override per-request up to TOP_K_MAX.
    # Used by: services/rag_pipeline.py
    TOP_K_DEFAULT: int = 5

    # Hard ceiling on top_k - enforced in request validation.
    # Prevents context window abuse and unbounded latency.
    # Used by: app/models/schemas.py (validator), services/rag_pipeline.py
    TOP_K_MAX: int = 10

    # FAISS fetch size when re-ranking is enabled.
    # Wider net for FAISS -> cross-encoder narrows to TOP_K_DEFAULT.
    # Used by: services/rag_pipeline.py
    TOP_K_RERANK_FETCH: int = 10

    # Minimum cosine similarity for a chunk to be included in context.
    # Chunks below this threshold are discarded before LLM call.
    # If zero chunks survive, pipeline returns early - no LLM call.
    # Used by: services/rag_pipeline.py
    SIMILARITY_THRESHOLD: float = 0.75

    # -- Re-ranking ----------------------------------------------------------
    # HuggingFace cross-encoder model for two-stage retrieval.
    # 22M params. CPU inference ~50-100ms for 10 candidates.
    # Must be preloaded at startup - see lifespan in main.py.
    # Used by: services/reranker.py
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Global default for re-ranking. Overridden per-request via
    # the `rerank` field in QueryRequest.
    # False by default to preserve baseline latency.
    # Used by: services/rag_pipeline.py
    RERANK_ENABLED: bool = False

    # -- Rate Limiting -------------------------------------------------------
    # Maximum requests per RATE_LIMIT_WINDOW_SECONDS per API key.
    # Token bucket capacity equals this value.
    # Used by: middleware/rate_limit.py, services/cache.py (Lua script)
    RATE_LIMIT_REQUESTS: int = 100

    # Rolling window duration in seconds for rate limit refill.
    # Bucket refills at RATE_LIMIT_REQUESTS per this window.
    # Used by: middleware/rate_limit.py, services/cache.py (Lua script)
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # -- Auth ----------------------------------------------------------------
    # Required. No default. Missing = ValidationError on startup.
    # Used to authenticate POST/GET/DELETE /admin/keys endpoints.
    # MUST be compared with hmac.compare_digest - never with ==.
    # Used by: middleware/auth.py
    ADMIN_SECRET: str

    # -- Application ---------------------------------------------------------
    # Semantic version string. Surfaced in GET /health response.
    # Used by: app/api/routes/health.py
    APP_VERSION: str = "1.1.0"

    # Logging level for the application. Valid values:
    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    # Used by: app/core/logging.py
    LOG_LEVEL: str = "INFO"

    # Runtime environment. Valid values: development, staging, production
    # Used for environment-aware behaviour and log formatting.
    # Used by: app/core/logging.py, app/main.py
    APP_ENV: str = "development"

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, value: object) -> str:
        if not isinstance(value, str):
            valid_values = ", ".join(ALLOWED_LOG_LEVELS)
            raise ValueError(f"LOG_LEVEL must be one of: {valid_values}.")

        normalized_value = value.upper()
        if normalized_value not in ALLOWED_LOG_LEVELS:
            valid_values = ", ".join(ALLOWED_LOG_LEVELS)
            raise ValueError(
                f"LOG_LEVEL must be one of: {valid_values}. "
                f"Got: {value!r}."
            )
        return normalized_value

    @field_validator("APP_ENV", mode="before")
    @classmethod
    def validate_app_env(cls, value: object) -> str:
        if not isinstance(value, str):
            valid_values = ", ".join(ALLOWED_APP_ENVS)
            raise ValueError(f"APP_ENV must be one of: {valid_values}.")

        normalized_value = value.lower()
        if normalized_value not in ALLOWED_APP_ENVS:
            valid_values = ", ".join(ALLOWED_APP_ENVS)
            raise ValueError(
                f"APP_ENV must be one of: {valid_values}. "
                f"Got: {value!r}."
            )
        return normalized_value

    @field_validator("SIMILARITY_THRESHOLD")
    @classmethod
    def validate_similarity_threshold(cls, value: float) -> float:
        if not SIMILARITY_THRESHOLD_MIN < value < SIMILARITY_THRESHOLD_MAX:
            raise ValueError(
                "SIMILARITY_THRESHOLD must be strictly between 0.0 and 1.0."
            )
        return value

    @model_validator(mode="after")
    def validate_chunk_overlap_tokens(self) -> Self:
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_SIZE_TOKENS:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be strictly less than "
                "CHUNK_SIZE_TOKENS."
            )
        return self


# Application code must use get_settings(); importing Settings directly is
# not part of the supported configuration access pattern.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.
    Call get_settings.cache_clear() in tests before reloading env state."""
    return Settings()


# Validate configuration during module import so startup fails fast.
get_settings()
