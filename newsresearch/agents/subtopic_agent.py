"""Subtopic Agent (Phase 2, Story 2.2, TRD's `agents/subtopic_agent.py`).

This module is populated incrementally across Phase 2's Wave 1-3 tasks:
propose candidate subtopics (Task 2.2.1, LLM-driven), broad topic-scoped
fetch (this file's `broad_topic_fetch`, Task 2.2.2), embed+cluster+reconcile
the broad set (Task 2.2.3), and rank/cap/excess-retention (Task 2.2.4).

Only `broad_topic_fetch` (Task 2.2.2) is implemented here so far -- the rest
is deliberately out of scope for this task (candidate proposal, clustering,
reconciliation, and ranking are separate `data-scientist`-designed,
`backend-engineer`-wired Wave 2/3 tasks per EXECUTION_PLAN.md's Phase 2 Role
Ownership & Parallelization Re-analysis).
"""

from __future__ import annotations

from typing import Any

from psycopg_pool import ConnectionPool

from newsresearch.agents.sourcing_agent import sourcing_agent
from newsresearch.config import Settings


def broad_topic_fetch(
    topic: str,
    lookback_days: int,
    *,
    pool: ConnectionPool | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Broad, topic-scoped fetch reusing Phase 1's `sourcing_agent` (Task
    2.2.2) -- not yet subtopic-filtered, since subtopics aren't known this
    early in the pipeline (that filtering happens once Task 2.2.1's
    candidate proposal and Task 2.2.3's reconciliation exist).

    The `topic` string itself is used as a single keyword -- deliberately no
    tokenization/keyword-expansion logic here, since a broad single-keyword
    query against `sourcing_agent`'s GDELT+RSS(+backfill) sources is already
    broad enough to surface multiple candidate-subtopic angles (e.g.
    "renewable energy" plausibly surfacing solar/wind/policy/storage
    articles), and inventing a smarter keyword-derivation strategy would be
    a query-design decision outside this task's scope.

    Returns raw article dicts (not `ScoredArticle` wrappers) -- downstream
    embedding/clustering (Task 2.2.3) operates on article text/title, not
    reputation scores.
    """
    scored_articles = sourcing_agent([topic], lookback_days, pool=pool, settings=settings)
    return [scored.article for scored in scored_articles]
