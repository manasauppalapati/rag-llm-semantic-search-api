"""
Document ingestion route handlers for upload and job dispatch.

When implemented, this module will validate multipart upload requests,
dispatch background ingestion work, and expose ingestion job status APIs.
Phase 5 — API layer.
Depends on: app.api.dependencies, app.models.schemas,
app.services.document_loader, app.services.chunker, app.services.embeddings,
app.services.vector_store, app.services.cache.
Depended on by: app.main.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 5")
