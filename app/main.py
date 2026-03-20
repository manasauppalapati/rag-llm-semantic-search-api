"""
Application factory for the RAG LLM Semantic Search API.

Creates and configures the FastAPI application instance,
registers middleware in the correct order (auth before
rate limiting), mounts all routers, and defines the lifespan
context manager responsible for startup preloading of FAISS
indexes and the cross-encoder re-ranking model.

Phase 6 — API layer.
Depends on: all services, all middleware, all routes.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown lifecycle.

    Startup:
        - Configures structured logging
        - Preloads all FAISS indexes from disk into memory
        - Preloads the cross-encoder re-ranking model
        - Verifies Redis connectivity

    Shutdown:
        - Flushes any pending writes
    """
    setup_logging()
    # Phase 6: vector_store.preload_all_namespaces()
    # Phase 6: reranker.preload()
    # Phase 6: await cache.ping()
    yield
    # Phase 6: graceful shutdown


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        Configured FastAPI app with middleware and routers mounted.
    """
    settings = get_settings()

    app = FastAPI(
        title="RAG LLM Semantic Search API",
        version=settings.APP_VERSION,
        description=(
            "Retrieval-Augmented Generation API with semantic search, "
            "namespace-isolated FAISS vector store, Redis caching, "
            "and optional cross-encoder re-ranking."
        ),
        lifespan=lifespan,
    )

    # Phase 5: app.add_middleware(RateLimitMiddleware)
    # Phase 5: app.add_middleware(AuthMiddleware)
    # Phase 6: app.include_router(health_router)
    # Phase 6: app.include_router(ingest_router)
    # Phase 6: app.include_router(query_router)
    # Phase 6: app.include_router(admin_router)

    return app


app = FastAPI()  # temporary — replaced in Phase 6
