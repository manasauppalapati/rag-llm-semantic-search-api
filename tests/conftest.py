"""
Shared pytest fixtures for the RAG LLM Semantic Search API test suite.

Fixtures defined here are available to all test modules without
explicit import. This file owns the test infrastructure:
mock Redis, mock OpenAI client, temporary FAISS directory,
mock re-ranker, and the FastAPI AsyncClient.

Phase 8 — Tests.
"""

# Phase 8: import pytest, fakeredis, httpx AsyncClient
# Phase 8: client fixture — httpx.AsyncClient wrapping create_app()
# Phase 8: mock_openai fixture — patches AsyncOpenAI
# Phase 8: fake_redis fixture — fakeredis.aioredis.FakeRedis
# Phase 8: tmp_faiss_dir fixture — tmp_path with .index files
# Phase 8: mock_reranker fixture — patches RerankerService.rerank
