"""
Vector store service for namespaced FAISS persistence and search.

When implemented, this module will own namespace locking, index preload,
vector writes, similarity search, and disk persistence.
Phase 2 — Storage.
Depends on: app.core.config, app.core.logging.
Depended on by: app.api.routes.health, app.api.routes.ingest,
app.services.rag_pipeline, app.main.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 2")
