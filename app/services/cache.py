"""
Caching service for Redis-backed answers, sessions, and job state.

When implemented, this module will centralize Redis access patterns, TTL
policy, session history storage, and ingestion job tracking.
Phase 2 — Storage.
Depends on: app.core.config, app.core.logging.
Depended on by: app.api.routes.ingest, app.api.routes.query,
app.api.routes.health, app.middleware.rate_limit, app.services.embeddings,
app.services.query_rewriter, app.services.rag_pipeline, app.main.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 2")
