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
from langgraph.types import Command
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
from newsresearch.sourcing.gdelt import GDELTError

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


def _render_gate1(value: dict) -> None:
    typer.echo("\n=== Gate 1: Subtopic Candidates ===")
    for i, candidate in enumerate(value.get("candidates", [])):
        typer.echo(f"  [{i}] {candidate['label']} (articles={candidate.get('article_count', '?')})")
    excess = value.get("excess") or []
    if excess:
        typer.echo("--- Also detected (excess) ---")
        for candidate in excess:
            typer.echo(f"  - {candidate['label']}")


def _prompt_gate1(value: dict) -> dict:
    """Render Gate 1's candidate/excess payload and prompt for approve/edit.

    Edit keeps this minimal (throwaway dev UX per `CLAUDE.md`): pick which
    candidates to keep by index, dropping the rest. `gate1_node`'s `edit`
    action only needs each surviving candidate's `label` to re-trigger real
    reconciliation (`make_real_reconcile`), so passing the kept candidate
    dicts through unchanged is enough.
    """
    candidates = value.get("candidates", [])
    _render_gate1(value)
    action = typer.prompt("Approve as-is or edit? [a/e]", default="a").strip().lower()
    if not action.startswith("e"):
        return {"action": "approve"}

    keep = typer.prompt(
        "Comma-separated indices to KEEP (blank = keep all)", default=""
    ).strip()
    if keep:
        keep_indices = {int(i) for i in keep.split(",")}
        edited = [c for i, c in enumerate(candidates) if i in keep_indices]
    else:
        edited = candidates
    return {"action": "edit", "candidates": edited}


def _prompt_gate2(cluster_report: dict) -> dict:
    """Render a per-subtopic Gate 2 cluster report and prompt to continue."""
    typer.echo("\n=== Gate 2: Cluster Report ===")
    typer.echo(f"cluster_sizes={cluster_report.get('cluster_sizes')}")
    typer.echo(f"source_spread={cluster_report.get('source_spread')}")
    for headline in cluster_report.get("sample_headlines", []):
        typer.echo(f"  - {headline}")
    typer.prompt("Press enter to continue", default="", show_default=False)
    return {"action": "continue"}


def _resume_payloads(interrupts) -> dict:
    """Build one resume payload per pending interrupt, keyed by interrupt id.

    Works uniformly for Gate 1 (a single pending interrupt) and Gate 2's N
    concurrent per-subtopic interrupts (Task 2.4.1's fan-out): LangGraph
    only requires interrupt-id keys once more than one interrupt is pending
    (`langgraph.pregel._loop`), so a single-entry id-keyed map resumes Gate 1
    just as well as an unkeyed `{"action": ...}` would.
    """
    payloads = {}
    for intr in interrupts:
        value = intr.value
        if "candidates" in value:
            payloads[intr.id] = _prompt_gate1(value)
        elif "cluster_report" in value:
            payloads[intr.id] = _prompt_gate2(value["cluster_report"])
        else:
            raise ValueError(f"unrecognized interrupt payload: {value!r}")
    return payloads


def _drain_interrupts(graph, result: dict, config: dict) -> dict:
    """Resume every currently-pending interrupt, looping until none remain.

    Gate 1 interrupts unconditionally after the real `subtopic` node
    populates `candidates`/`excess`; approving/editing there fans out into
    one independent Gate 2 interrupt per approved subtopic (Task 2.4.1).
    Each pass renders every currently-pending interrupt (Gate 1's single
    one, then Gate 2's N per-subtopic ones) and resumes them all in one
    `Command(resume=...)` call, per-branch. Shared between the fresh-run and
    `--thread-id` resume paths (Task 2.8.3) since both reach this same
    loop, just from a different starting `result`.
    """
    while "__interrupt__" in result:
        result = graph.invoke(
            Command(resume=_resume_payloads(result["__interrupt__"])), config=config
        )
    return result


@app.command()
def run(
    topic: str = typer.Argument(..., help="Topic string to research."),
    thread_id: str = typer.Option(
        None,
        "--thread-id",
        help=(
            "Rejoin a previously interrupted run by its thread_id/run_id "
            "(printed at the start of every run) instead of starting a new "
            "one. TOPIC is still required by the CLI but is ignored -- the "
            "resumed run's own persisted topic is used."
        ),
    ),
) -> None:
    """Run the Phase-0 no-op graph end-to-end for TOPIC, with observability attached."""
    settings = Settings()
    if not settings.database_url:
        raise typer.BadParameter(
            "NEWSRESEARCH_DATABASE_URL must be set (app Postgres) to run the graph."
        )

    pool = init_db(settings.database_url)
    cost_callback = CostCallbackHandler(pool)
    langfuse_callback = get_langfuse_callback_handler(settings)
    graph = build_graph(database_url=settings.database_url)

    if thread_id:
        run_id = thread_id
        config = {
            "configurable": {"thread_id": run_id},
            "callbacks": [cost_callback, langfuse_callback],
            "metadata": {**trace_metadata(run_id), "stage": "subtopic"},
        }
        snapshot = graph.get_state(config)
        pending_interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if not pending_interrupts:
            raise typer.BadParameter(
                f"No pending interrupt found for thread_id={run_id!r} -- it may "
                "not exist, or the run may have already completed."
            )
        typer.echo(f"run_id={run_id} (resuming)")
        with mlflow_run(
            run_id,
            params={"topic": topic, "max_subtopics": settings.pipeline.max_subtopics},
            settings=settings,
        ):
            result = graph.invoke(
                Command(resume=_resume_payloads(pending_interrupts)), config=config
            )
            result = _drain_interrupts(graph, result, config)
    else:
        run_id = f"run-{uuid.uuid4()}"
        _record_run(pool, run_id)
        typer.echo(f"run_id={run_id} (starting) -- pass --thread-id={run_id} to resume if killed")
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
            # The real `subtopic` node (Story 2.2 production wiring) makes the
            # only LLM call before Gate 1; per-node stage tagging beyond that is
            # a Phase 2+ concern for later nodes as they land.
            "metadata": {**trace_metadata(run_id), "stage": "subtopic"},
        }

        with mlflow_run(
            run_id,
            params={"topic": topic, "max_subtopics": settings.pipeline.max_subtopics},
            settings=settings,
        ):
            result = graph.invoke(initial_state, config=config)
            result = _drain_interrupts(graph, result, config)

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
    except GDELTError as exc:
        # Story 1.12 made `sourcing_agent` soft-fail GDELT (log a warning,
        # continue with RSS(+backfill)-only results) -- so this branch is
        # now unreachable for the GDELT case in normal operation. Left in
        # place as defensive dead code (not deleted) rather than assuming
        # with full confidence that no other path could ever raise
        # `GDELTError` through this CLI command.
        typer.echo(f"GDELT error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
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
