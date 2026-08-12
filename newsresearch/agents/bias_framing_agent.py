"""Bias & Framing Agent (Phase 4, Story 4.1, Task 4.1.1b).

Wires the data-scientist-authored `llm/prompts/bias_framing.txt` (Task
4.1.1a, see `notebooks/phase4_bias_framing_prompt_review.md` on
`feat/datascientist`) through `get_chat_model("bias_framing")` with
`.with_structured_output(BiasFramingBatch)`, per
`agents/summarization_agent.py`'s established pattern.

Reads one subtopic's already-persisted claim clusters (Task 3.5.1) and
batches them `Settings.models.bias_framing_batch_size` clusters per call
(tech-lead, 2026-08-12), merging each batch's `clusters` list additively
into the subtopic's result set. Only asserting rows are sent -- an
`omits` row is not a claim, and omission is read deterministically from
`claim_cluster_articles` downstream, never asked of the model.

Returns the structured output; persisting it is Task 4.4.1's job.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.llm.schemas import BiasFramingBatch, ClusterFraming
from newsresearch.observability.langfuse_setup import (
    get_langfuse_callback_handler,
    trace_metadata,
)
from newsresearch.persistence.claim_clusters import (
    read_cluster_article_relations,
    read_subtopic_cluster_ids,
)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def _format_batch(batch: list[tuple[str, list[dict]]]) -> str:
    """Matches Task 4.1.1a's own `phase4_bias_framing_review.py::format_batch`.

    The "Distinct articles" roster is computed here and given to the model
    as fact. It is load-bearing, not decoration: 4.1.1a measured that
    making the model derive it instead regressed `single_article_only`
    accuracy from 0/69 to 6/69, because the model reads one *domain* with
    two articles as one article (the issue #114 failure class).
    """
    parts = []
    for cluster_id, rows in batch:
        distinct = list(dict.fromkeys(r["article_id"] for r in rows))
        lines = [
            f"Cluster {cluster_id}",
            f"Distinct articles in this cluster ({len(distinct)}): " + ", ".join(distinct),
            "Claims (article id | source domain | claim):",
        ]
        lines += [f"- {r['article_id']} | {r['domain']} | {r['claim_text']}" for r in rows]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def analyze_subtopic_framing(
    pool: ConnectionPool,
    subtopic_id: str,
    *,
    run_id: str = "dev",
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> list[ClusterFraming]:
    """Describe how each article frames each of one subtopic's claim clusters.

    Traced via Langfuse per `summarize_cluster`'s established convention,
    tagged `stage=bias_framing`. Accepts an ambient `config` and merges it
    via `merge_configs` rather than replacing it, so nested callbacks (e.g.
    `cost_callback.py`) still fire.
    """
    settings = settings or Settings()

    clusters: list[tuple[str, list[dict]]] = []
    for cluster_id in read_subtopic_cluster_ids(pool, subtopic_id):
        rows = [r for r in read_cluster_article_relations(pool, cluster_id) if r["relation"] == "asserts"]
        if rows:
            clusters.append((cluster_id, rows))
    if not clusters:
        return []

    template_text = (_PROMPTS_DIR / "bias_framing.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("bias_framing").with_structured_output(BiasFramingBatch)
    chain = prompt | model

    call_config: RunnableConfig = {
        "callbacks": [get_langfuse_callback_handler(settings)],
        "metadata": {**trace_metadata(run_id, subtopic_id), "stage": "bias_framing"},
    }
    merged_config = merge_configs(config, call_config)

    batch_size = settings.models.bias_framing_batch_size
    results: list[ClusterFraming] = []
    for start in range(0, len(clusters), batch_size):
        batch = clusters[start : start + batch_size]
        result = chain.invoke({"clusters": _format_batch(batch)}, config=merged_config)
        results.extend(result.clusters)
    return results
