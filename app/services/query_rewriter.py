"""
Query rewriting service for ambiguous multi-turn follow-up requests.

When implemented, this module will decide when a query needs rewriting and
call the completion model to produce a self-contained search query.
Phase 4 — Intelligence.
Depends on: app.core.config, app.core.logging, app.services.cache.
Depended on by: app.services.rag_pipeline.
"""


def _placeholder() -> None:
    raise NotImplementedError("Implementation pending — Phase 4")
