"""
Embeddings service for generating vector representations asynchronously.

When implemented, this module will batch OpenAI embedding calls, deduplicate
by content hash, and return vectors for ingest and query flows.
Phase 3 — Processing.
Depends on: app.core.config, app.core.logging, app.services.cache.
Depended on by: app.api.routes.ingest, app.services.rag_pipeline.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 3")
