"""Summarization Agent (Phase 3, Story 3.6, Task 3.6.1b).

Wires the data-scientist-authored `llm/prompts/summarization.txt` (Task
3.6.1a, see `notebooks/phase3_summarization_prompt_review.md` on
`feat/datascientist`) through `get_chat_model("summarization")`, per
`agents/claim_extraction_agent.py`'s established pattern. Plain string
output -- `claim_clusters.summary` is `TEXT`, no schema per 3.6.1a's design
decision.

One call per persisted claim cluster (Task 3.5.1's `persistence/
claim_clusters.py`): reads the cluster's asserting claims back via
`read_cluster_article_relations`, writes the summary back via
`write_cluster_summary`. Omitting rows (claim_text NULL) are not claims and
are excluded from the prompt input.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.observability.langfuse_setup import get_langfuse_callback_handler, trace_metadata
from newsresearch.persistence.claim_clusters import read_cluster_article_relations, write_cluster_summary

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def _format_claims(rows: list[dict]) -> str:
    """Matches Task 3.6.1a's own `phase3_summarization_review.py::format_claims`."""
    return "\n".join(f"- {r['domain']}: {r['claim_text']}" for r in rows if r["relation"] == "asserts")


def summarize_cluster(
    pool: ConnectionPool,
    cluster_id: str,
    *,
    run_id: str = "dev",
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> str:
    """Summarize one persisted claim cluster and write the result back.

    Traced via Langfuse per `extract_claims`'s established convention, tagged
    `stage=summarization`. Accepts an ambient `config` and merges it via
    `merge_configs` rather than replacing it, so nested callbacks (e.g.
    `cost_callback.py`) still fire.
    """
    settings = settings or Settings()

    rows = read_cluster_article_relations(pool, cluster_id)
    claims_text = _format_claims(rows)

    template_text = (_PROMPTS_DIR / "summarization.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("summarization")
    chain = prompt | model

    call_config: RunnableConfig = {
        "callbacks": [get_langfuse_callback_handler(settings)],
        "metadata": {**trace_metadata(run_id), "stage": "summarization"},
    }
    merged_config = merge_configs(config, call_config)
    result = chain.invoke({"claims": claims_text}, config=merged_config)

    summary = result.content
    write_cluster_summary(pool, cluster_id, summary)
    return summary