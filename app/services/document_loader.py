"""
Document loading service for PDF and plain-text ingestion sources.

When implemented, this module will normalize supported document formats into
ordered page text suitable for chunking and downstream processing.
Phase 3 — Processing.
Depends on: uploaded files, app.core.logging.
Depended on by: app.api.routes.ingest, app.services.chunker.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 3")
