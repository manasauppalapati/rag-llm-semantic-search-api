"""
FastAPI dependency providers for shared application services.

When implemented, this module will expose Depends() helpers for cache,
vector store, embeddings, reranker, and RAG pipeline instances shared across
route handlers.
Phase 5 — API layer.
Depends on: app.services.cache, app.services.vector_store,
app.services.embeddings, app.services.reranker, app.services.rag_pipeline.
Depended on by: app.api.routes.ingest, app.api.routes.query,
app.api.routes.admin, app.api.routes.health.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 5")
