"""Subtopic Agent (Phase 2, Story 2.2, TRD's `agents/subtopic_agent.py`).

This module is populated incrementally across Phase 2's Wave 1-3 tasks:
propose candidate subtopics (this file's `propose_candidates`, Task 2.2.1b,
LLM-driven), broad topic-scoped fetch (`broad_topic_fetch`, Task 2.2.2),
embed+cluster+reconcile the broad set (Task 2.2.3), and rank/cap/
excess-retention (Task 2.2.4).

`propose_candidates` (Task 2.2.1b) and `broad_topic_fetch` (Task 2.2.2) are
implemented here so far -- clustering/reconciliation/ranking are separate
`data-scientist`-designed, `backend-engineer`-wired Wave 2/3 tasks per
EXECUTION_PLAN.md's Phase 2 Role Ownership & Parallelization Re-analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from psycopg_pool import ConnectionPool

from newsresearch.agents.sourcing_agent import sourcing_agent
from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.llm.schemas import SubtopicCandidateList
from newsresearch.observability.langfuse_setup import get_langfuse_callback_handler, trace_metadata

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def propose_candidates(
    topic: str,
    n_candidates: int = 8,
    *,
    run_id: str = "dev",
    settings: Settings | None = None,
) -> SubtopicCandidateList:
    """Propose `n_candidates` candidate subtopics for `topic` (Task 2.2.1b).

    Wires the data-scientist-authored `llm/prompts/subtopic_propose.txt`
    (Task 2.2.1a, see `notebooks/phase2-subtopic-prompt-design.md`) through
    `get_chat_model("subtopic").with_structured_output(SubtopicCandidateList)`
    per NFR-6's single-factory convention. `n_candidates` defaults to 8 per
    the design doc's recommendation (`Settings.pipeline.max_subtopics + 3`
    given the default `max_subtopics=5`) -- not itself a `Settings` field
    today, left as a caller-supplied parameter per the prompt design's intent
    that this is a starting recommendation, not a tuned config value.

    The returned `candidates` list may be shorter than `n_candidates` (the
    prompt's rule 6 lets the model propose fewer rather than pad with
    near-duplicates/filler) -- not treated as an error here.

    Traced via Langfuse per the CLI's established `get_langfuse_callback_handler`
    + `trace_metadata` convention, tagged with `run_id` and `stage=subtopic`.
    """
    settings = settings or Settings()

    template_text = (_PROMPTS_DIR / "subtopic_propose.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("subtopic").with_structured_output(SubtopicCandidateList)
    chain = prompt | model

    config = {
        "callbacks": [get_langfuse_callback_handler(settings)],
        "metadata": {**trace_metadata(run_id), "stage": "subtopic"},
    }
    return chain.invoke({"topic": topic, "n_candidates": n_candidates}, config=config)


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
