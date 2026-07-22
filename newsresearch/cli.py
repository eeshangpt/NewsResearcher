"""Typer CLI dev harness (Phase 0 Task 0.8.1 + Task 0.7.4).

`uv run newsresearch run "<topic>"` drives the compiled no-op graph
(`graph.build.build_graph()`) end-to-end. All three Story 0.7 observability
integrations are attached at this single top-level `graph.invoke()` call, per
Cross-Cutting Concerns: the cost callback (writes `run_costs` rows to
Postgres), the Langfuse callback (traces every LLM call, tagged with
`run_id`), and an MLflow run wrapping the whole invocation.

Throwaway-quality UX per EXECUTION_PLAN.md's "Dev-time pipeline harness" --
replaced by Streamlit in Phase 6, load-bearing for manual validation of every
phase between.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import typer
from psycopg_pool import ConnectionPool

from newsresearch.agents.sourcing_agent import sourcing_agent
from newsresearch.config import Settings
from newsresearch.graph.build import build_graph
from newsresearch.graph.state import GraphState
from newsresearch.observability.cost_callback import CostCallbackHandler
from newsresearch.observability.langfuse_setup import (
    get_langfuse_callback_handler,
    trace_metadata,
)
from newsresearch.observability.mlflow_setup import mlflow_run
from newsresearch.persistence.db import init_db

app = typer.Typer(help="NewsResearch dev CLI harness.")
dev_app = typer.Typer(help="Manual dev-inspection subcommands (Story 1.10).")
app.add_typer(dev_app, name="dev")


@app.callback()
def _callback() -> None:
    """NewsResearch dev CLI harness.

    An explicit (no-op) callback keeps `run` addressable as a named
    subcommand (`newsresearch run "<topic>"`) -- Typer otherwise collapses a
    single-command app into a bare top-level command with no subcommand name.
    """


def _record_run(pool: ConnectionPool, run_id: str) -> None:
    """Insert the `runs` row that `run_costs.run_id`'s FK requires.

    `persistence/queries.py` doesn't exist yet (a later task) -- this is the
    minimal write the CLI itself needs so its own `run_id` is valid for
    `cost_callback.py` to attach `run_costs` rows to.
    """
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, run_type, started_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (run_id, "manual", datetime.now(timezone.utc)),
        )


@app.command()
def run(topic: str = typer.Argument(..., help="Topic string to research.")) -> None:
    """Run the Phase-0 no-op graph end-to-end for TOPIC, with observability attached."""
    settings = Settings()
    if not settings.database_url:
        raise typer.BadParameter(
            "NEWSRESEARCH_DATABASE_URL must be set (app Postgres) to run the graph."
        )

    run_id = f"run-{uuid.uuid4()}"
    pool = init_db(settings.database_url)
    _record_run(pool, run_id)

    cost_callback = CostCallbackHandler(pool)
    langfuse_callback = get_langfuse_callback_handler(settings)

    graph = build_graph(database_url=settings.database_url)
    initial_state: GraphState = {
        "topic": topic,
        "canonical_topic": topic.strip().lower(),
        "run_id": run_id,
        "subtopics": [],
        "approved": False,
    }
    config = {
        "configurable": {"thread_id": run_id},
        "callbacks": [cost_callback, langfuse_callback],
        # Phase 0 only makes one stub LLM call (the `subtopic` node, Task
        # 0.7.4) so a single "stage" for the whole invocation is accurate;
        # per-node stage tagging is a Phase 1+ concern once real agents exist.
        "metadata": {**trace_metadata(run_id), "stage": "subtopic"},
    }

    with mlflow_run(
        run_id,
        params={"topic": topic, "max_subtopics": settings.pipeline.max_subtopics},
        settings=settings,
    ):
        result = graph.invoke(initial_state, config=config)

    typer.echo(f"run_id={run_id} topic={result['topic']!r} completed.")


@dev_app.command("sourcing-test")
def sourcing_test(
    keywords: str = typer.Argument(
        ..., help='Whitespace-separated keywords, e.g. "climate policy" -> ["climate", "policy"].'
    ),
    lookback_days: int = typer.Option(7, help="How many days back to search."),
) -> None:
    """Run `agents/sourcing_agent.py` against real GDELT/RSS and print each
    resulting article's URL, domain, reputation score, and tier.

    Manual dev-inspection harness (Story 1.10) -- not part of the LangGraph
    pipeline itself, just a thin wrapper for spot-checking sourcing +
    reputation scoring against real APIs during development.
    """
    settings = Settings()
    if not settings.database_url:
        raise typer.BadParameter(
            "NEWSRESEARCH_DATABASE_URL must be set (app Postgres) for the reputation cache."
        )

    pool = init_db(settings.database_url)
    try:
        results = sourcing_agent(keywords.split(), lookback_days, pool=pool, settings=settings)
    finally:
        pool.close()

    if not results:
        typer.echo("No articles survived sourcing + reputation filtering.")
        return

    for scored in results:
        article = scored.article
        typer.echo(
            f"{article['url']} | domain={article['domain']} "
            f"score={scored.reputation_score:.2f} tier={scored.reputation_tier}"
        )


if __name__ == "__main__":
    app()
