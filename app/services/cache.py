"""
Redis storage layer for the RAG LLM Semantic Search API.

Responsibilities:
    Own every direct Redis operation used by caching, sessions,
    ingestion job tracking, document hash registration, API key
    storage, and token-bucket rate limiting.

Why this module exists:
    Keeping Redis access in one place prevents key drift, mixed
    serialisation formats, inconsistent TTL handling, and ad hoc
    error handling spread across the codebase.

Failure model:
    Redis is allowed to degrade without taking the API down.
    Connection and timeout failures log with context and return
    cache misses, empty collections, or allowed rate-limit results
    instead of surfacing as request-breaking exceptions.

Concurrency model:
    All Redis access uses redis.asyncio exclusively. No sync Redis
    client is imported anywhere in this module.
"""

from datetime import datetime, timezone
import json
import time
from typing import TypedDict

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.logging import get_logger


_KEY_ANSWER: str = "cache:answer:{}"
_KEY_SESSION: str = "session:{}"
_KEY_JOB: str = "job:{}"
_KEY_DOC_HASH: str = "doc:hash:{}"
_KEY_API_KEY: str = "api_key:{}"
_KEY_RATE: str = "rate:{}"
_API_KEY_SCAN_PATTERN: str = _KEY_API_KEY.format("*")
_DOC_HASH_PRESENT_VALUE: str = "1"
_REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: int = 5
_REDIS_SOCKET_TIMEOUT_SECONDS: int = 5
_RATE_LIMIT_RESULT_LENGTH: int = 2
_RATE_LIMIT_ALLOWED_RESULT: int = 1
_RATE_LIMIT_DENIED_RESULT: int = 0
_RATE_LIMIT_RETRY_AFTER_ALLOWED: int = 0
_RATE_LIMIT_LUA_SCRIPT: str = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local window = tonumber(ARGV[3])
local bucket = redis.call('GET', key)
local tokens = capacity
local last_refill = now
if bucket then
  local data = cjson.decode(bucket)
  local elapsed = now - data['last_refill']
  local refill = (elapsed / window) * capacity
  tokens = math.min(capacity, data['tokens'] + refill)
  last_refill = data['last_refill']
end
if tokens < 1 then
  local retry = math.ceil(window - (now - last_refill))
  return {0, retry}
end
local new_bucket = cjson.encode(
  {tokens=tokens-1, last_refill=last_refill}
)
redis.call(
  'SETEX', key,
  tonumber(ARGV[4]),
  new_bucket
)
return {1, 0}
"""

logger = get_logger(__name__)


class SessionTurn(TypedDict):
    """Single conversation turn stored in Redis session history."""

    role: str
    content: str


class JobRecord(TypedDict, total=False):
    """JSON-serialisable ingestion job state stored in Redis."""

    status: str
    chunks_indexed: int
    files_processed: int
    errors: list[str]
    completed_at: str | None


class ApiKeyRecord(TypedDict, total=False):
    """JSON-serialisable API key metadata stored in Redis."""

    key_id: str
    name: str
    created_at: str
    last_used_at: str | None


def _utc_now_isoformat() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_session_history(payload: str) -> list[SessionTurn] | None:
    """Parse a session history JSON payload into typed turns."""
    raw_value: object = json.loads(payload)
    if not isinstance(raw_value, list):
        return None

    history: list[SessionTurn] = []
    for item in raw_value:
        if not isinstance(item, dict):
            return None

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        history.append({"role": role, "content": content})
    return history


def _parse_job_record(payload: str) -> JobRecord | None:
    """Parse a job-state JSON payload into a typed record."""
    raw_value: object = json.loads(payload)
    if not isinstance(raw_value, dict):
        return None

    job_record: JobRecord = {}

    status = raw_value.get("status")
    if isinstance(status, str):
        job_record["status"] = status

    chunks_indexed = raw_value.get("chunks_indexed")
    if isinstance(chunks_indexed, int):
        job_record["chunks_indexed"] = chunks_indexed

    files_processed = raw_value.get("files_processed")
    if isinstance(files_processed, int):
        job_record["files_processed"] = files_processed

    errors = raw_value.get("errors")
    if isinstance(errors, list) and all(isinstance(item, str) for item in errors):
        job_record["errors"] = errors

    completed_at = raw_value.get("completed_at")
    if completed_at is None or isinstance(completed_at, str):
        job_record["completed_at"] = completed_at

    return job_record


def _parse_api_key_record(payload: str) -> ApiKeyRecord | None:
    """Parse an API key JSON payload into a typed record."""
    raw_value: object = json.loads(payload)
    if not isinstance(raw_value, dict):
        return None

    key_record: ApiKeyRecord = {}

    key_id = raw_value.get("key_id")
    if isinstance(key_id, str):
        key_record["key_id"] = key_id

    name = raw_value.get("name")
    if isinstance(name, str):
        key_record["name"] = name

    created_at = raw_value.get("created_at")
    if isinstance(created_at, str):
        key_record["created_at"] = created_at

    last_used_at = raw_value.get("last_used_at")
    if last_used_at is None or isinstance(last_used_at, str):
        key_record["last_used_at"] = last_used_at

    return key_record


class CacheService:
    """Centralised async Redis service used by all storage-dependent code."""

    def __init__(self, redis_client: redis.Redis) -> None:
        """Store the Redis client and cached application settings."""
        self._redis = redis_client
        self._settings = get_settings()

    async def ping(self) -> bool:
        """
        Verify Redis connectivity without raising to the caller.

        Returns:
            True when the Redis server responds to PING, otherwise False.
        """
        try:
            await self._redis.ping()
        except RedisError:
            logger.warning("Redis ping failed; cache service is degraded.",
                           exc_info=True)
            return False

        logger.debug("Redis ping succeeded.")
        return True

    async def get_answer(self, embedding_hash: str) -> str | None:
        """
        Read a cached answer payload by semantic embedding hash.

        Returns:
            The cached JSON string on hit, or None on miss or Redis error.
        """
        key = _KEY_ANSWER.format(embedding_hash)
        try:
            cached_answer = await self._redis.get(key)
        except RedisError:
            logger.error("Failed to read answer cache entry.", exc_info=True)
            return None

        if cached_answer is None:
            logger.debug("Answer cache miss.", extra={"key": key})
            return None

        logger.debug("Answer cache hit.", extra={"key": key})
        return cached_answer

    async def set_answer(self, embedding_hash: str, answer_json: str) -> None:
        """
        Write a cached answer payload with the configured answer TTL.

        Returns:
            None.
        """
        key = _KEY_ANSWER.format(embedding_hash)
        try:
            await self._redis.setex(
                key,
                self._settings.REDIS_TTL_ANSWER,
                answer_json,
            )
        except RedisError:
            logger.error("Failed to write answer cache entry.", exc_info=True)
            return

        logger.debug("Stored answer cache entry.", extra={"key": key})

    async def get_session(self, session_id: str) -> list[SessionTurn] | None:
        """
        Read and refresh a conversation session history payload.

        Returns:
            The deserialised session history, or None on miss or error.
        """
        key = _KEY_SESSION.format(session_id)
        try:
            payload = await self._redis.get(key)
            if payload is None:
                logger.debug("Session cache miss.", extra={"session_id": session_id})
                return None

            history = _parse_session_history(payload)
            if history is None:
                logger.error("Session payload could not be parsed.")
                return None

            await self._redis.expire(key, self._settings.REDIS_TTL_SESSION)
        except (RedisError, json.JSONDecodeError, TypeError, ValueError):
            logger.error("Failed to read session history.", exc_info=True)
            return None

        logger.debug("Loaded session history.", extra={"session_id": session_id})
        return history

    async def set_session(
        self,
        session_id: str,
        history: list[SessionTurn],
    ) -> None:
        """
        Persist a conversation session history payload with TTL refresh policy.

        Returns:
            None.
        """
        key = _KEY_SESSION.format(session_id)
        try:
            payload = json.dumps(history)
            await self._redis.setex(
                key,
                self._settings.REDIS_TTL_SESSION,
                payload,
            )
        except (RedisError, TypeError, ValueError):
            logger.error("Failed to store session history.", exc_info=True)
            return

        logger.debug("Stored session history.", extra={"session_id": session_id})

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session history payload.

        Returns:
            True when a session key was deleted, otherwise False.
        """
        key = _KEY_SESSION.format(session_id)
        try:
            deleted_count = await self._redis.delete(key)
        except RedisError:
            logger.error("Failed to delete session history.", exc_info=True)
            return False

        deleted = deleted_count > 0
        logger.debug(
            "Deleted session history.",
            extra={"session_id": session_id, "deleted": deleted},
        )
        return deleted

    async def get_job(self, job_id: str) -> JobRecord | None:
        """
        Read ingestion job state from Redis.

        Returns:
            The deserialised job record, or None on miss or error.
        """
        key = _KEY_JOB.format(job_id)
        try:
            payload = await self._redis.get(key)
            if payload is None:
                logger.debug("Job state miss.", extra={"job_id": job_id})
                return None

            job_record = _parse_job_record(payload)
            if job_record is None:
                logger.error("Job payload could not be parsed.")
                return None
        except (RedisError, json.JSONDecodeError, TypeError, ValueError):
            logger.error("Failed to read job state.", exc_info=True)
            return None

        logger.debug("Loaded job state.", extra={"job_id": job_id})
        return job_record

    async def set_job(self, job_id: str, job_data: JobRecord) -> None:
        """
        Persist ingestion job state using the configured job TTL.

        Returns:
            None.
        """
        key = _KEY_JOB.format(job_id)
        try:
            payload = json.dumps(job_data)
            await self._redis.setex(
                key,
                self._settings.REDIS_TTL_JOB,
                payload,
            )
        except (RedisError, TypeError, ValueError):
            logger.error("Failed to store job state.", exc_info=True)
            return

        logger.debug("Stored job state.", extra={"job_id": job_id})

    async def is_doc_hash_known(self, content_hash: str) -> bool:
        """
        Check whether a content hash has already been registered.

        Returns:
            True when the hash exists in Redis, otherwise False.
        """
        key = _KEY_DOC_HASH.format(content_hash)
        try:
            exists = await self._redis.exists(key)
        except RedisError:
            logger.error("Failed to read document hash registry.", exc_info=True)
            return False

        is_known = exists > 0
        logger.debug(
            "Checked document hash registry.",
            extra={"content_hash": content_hash, "known": is_known},
        )
        return is_known

    async def register_doc_hash(self, content_hash: str) -> None:
        """
        Register a document content hash with no TTL for deduplication.

        Returns:
            None.
        """
        key = _KEY_DOC_HASH.format(content_hash)
        try:
            await self._redis.set(key, _DOC_HASH_PRESENT_VALUE)
        except RedisError:
            logger.error("Failed to register document hash.", exc_info=True)
            return

        logger.debug("Registered document hash.", extra={"content_hash": content_hash})

    async def get_api_key(self, key_hash: str) -> ApiKeyRecord | None:
        """
        Read API key metadata by hashed key value.

        Returns:
            The deserialised key record, or None on miss or error.
        """
        key = _KEY_API_KEY.format(key_hash)
        try:
            payload = await self._redis.get(key)
            if payload is None:
                logger.debug("API key record miss.")
                return None

            key_record = _parse_api_key_record(payload)
            if key_record is None:
                logger.error("API key payload could not be parsed.")
                return None
        except (RedisError, json.JSONDecodeError, TypeError, ValueError):
            logger.error("Failed to read API key record.", exc_info=True)
            return None

        logger.debug("Loaded API key record.")
        return key_record

    async def set_api_key(self, key_hash: str, key_data: ApiKeyRecord) -> None:
        """
        Persist API key metadata without a TTL.

        Returns:
            None.
        """
        key = _KEY_API_KEY.format(key_hash)
        try:
            payload = json.dumps(key_data)
            await self._redis.set(key, payload)
        except (RedisError, TypeError, ValueError):
            logger.error("Failed to store API key record.", exc_info=True)
            return

        logger.debug("Stored API key record.")

    async def delete_api_key(self, key_hash: str) -> bool:
        """
        Delete an API key record by hashed key value.

        Returns:
            True when a key record was deleted, otherwise False.
        """
        key = _KEY_API_KEY.format(key_hash)
        try:
            deleted_count = await self._redis.delete(key)
        except RedisError:
            logger.error("Failed to delete API key record.", exc_info=True)
            return False

        deleted = deleted_count > 0
        logger.debug("Deleted API key record.", extra={"deleted": deleted})
        return deleted

    async def update_last_used(self, key_hash: str) -> None:
        """
        Update last_used_at on an existing API key record.

        Returns:
            None.
        """
        key_record = await self.get_api_key(key_hash)
        if key_record is None:
            logger.warning(
                "Cannot update last_used_at because API key record is missing.",
                extra={"key_hash_present": False},
            )
            return

        updated_record: ApiKeyRecord = {
            **key_record,
            "last_used_at": _utc_now_isoformat(),
        }
        await self.set_api_key(key_hash, updated_record)
        logger.debug("Updated API key last_used_at timestamp.")

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        """
        List all API key metadata records using SCAN.

        Returns:
            A list of deserialised key records, or an empty list on error.
        """
        cursor = 0
        records: list[ApiKeyRecord] = []

        try:
            while True:
                next_cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match=_API_KEY_SCAN_PATTERN,
                )
                cursor = int(next_cursor)

                for key in keys:
                    payload = await self._redis.get(key)
                    if payload is None:
                        continue

                    key_record = _parse_api_key_record(payload)
                    if key_record is None:
                        logger.error(
                            "Encountered an invalid API key record during scan."
                        )
                        continue
                    records.append(key_record)

                if cursor == 0:
                    break
        except (RedisError, json.JSONDecodeError, TypeError, ValueError):
            logger.error("Failed to list API key records.", exc_info=True)
            return []

        logger.debug("Listed API key records.", extra={"count": len(records)})
        return records

    async def consume_token(self, key_id: str) -> tuple[bool, int]:
        """
        Consume one rate-limit token via an atomic Redis Lua script.

        Returns:
            A tuple of (allowed, retry_after_seconds).
        """
        key = _KEY_RATE.format(key_id)
        try:
            result = await self._redis.eval(
                _RATE_LIMIT_LUA_SCRIPT,
                1,
                key,
                self._settings.RATE_LIMIT_REQUESTS,
                int(time.time()),
                self._settings.RATE_LIMIT_WINDOW_SECONDS,
                self._settings.REDIS_TTL_RATE_BUCKET,
            )
            if not isinstance(result, (list, tuple)):
                logger.error("Rate limit script returned an invalid payload.")
                return True, _RATE_LIMIT_RETRY_AFTER_ALLOWED

            if len(result) != _RATE_LIMIT_RESULT_LENGTH:
                logger.error("Rate limit script returned an invalid length.")
                return True, _RATE_LIMIT_RETRY_AFTER_ALLOWED

            allowed = int(result[0]) == _RATE_LIMIT_ALLOWED_RESULT
            retry_after = int(result[1])
        except (RedisError, TypeError, ValueError):
            logger.error("Failed to consume a rate-limit token.", exc_info=True)
            return True, _RATE_LIMIT_RETRY_AFTER_ALLOWED

        if allowed:
            logger.debug("Rate limit token consumed.", extra={"key_id": key_id})
            return True, _RATE_LIMIT_RETRY_AFTER_ALLOWED

        logger.debug(
            "Rate limit bucket exhausted.",
            extra={"key_id": key_id, "retry_after": retry_after},
        )
        return False, retry_after


async def create_cache_service() -> CacheService:
    """
    Create a CacheService backed by redis.asyncio.from_url().

    Returns:
        A CacheService instance, even when Redis is currently degraded.
    """
    settings = get_settings()
    redis_client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=_REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=_REDIS_SOCKET_TIMEOUT_SECONDS,
    )
    service = CacheService(redis_client=redis_client)
    if not await service.ping():
        logger.warning("Cache service created in degraded mode.")
    return service
