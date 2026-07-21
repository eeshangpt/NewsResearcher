"""MLflow run lifecycle helpers (Cross-Cutting Concerns: pipeline/experiment-
level tracking, distinct from `langfuse_setup.py`'s raw LLM-call tracing).

One MLflow run per pipeline run, keyed by `run_id`, against a local
file-store backend (`Settings.mlflow_tracking_uri`, default `./mlruns`) --
zero extra infra. Logs a config snapshot as params and is the natural home
for the final snapshot JSON as an artifact once later phases produce one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking.fluent import ActiveRun

from newsresearch.config import Settings

EXPERIMENT_NAME = "newsresearch-pipeline-runs"


def _configure_tracking(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    # MLflow 3.x's plain filesystem backend (e.g. "./mlruns") is in
    # maintenance mode and raises unless explicitly opted back into -- the
    # documented escape hatch, since EXECUTION_PLAN.md's "zero extra infra"
    # file-store choice is a cross-cutting decision this module implements,
    # not one to silently swap for a database backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # Unlike a real server-backed tracking URI, the file store doesn't create
    # its own root directory ahead of first use on every code path -- so a
    # fresh clone's first run (or a scratch dir in tests) needs it created
    # explicitly rather than relying on FileStore to do it lazily.
    Path(settings.mlflow_tracking_uri).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(
    run_id: str, params: dict[str, Any] | None = None, settings: Settings | None = None
) -> ActiveRun:
    """Start exactly one MLflow run tagged with `run_id`.

    `params` (e.g. model choices, `max_subtopics`, reputation weights,
    clustering thresholds -- a config snapshot) is logged as MLflow params
    if given. Pair with `end_run()` when done, or use `mlflow_run()` below to
    get both for free.
    """
    _configure_tracking(settings)
    active_run = mlflow.start_run(run_name=run_id, tags={"run_id": run_id})
    if params:
        mlflow.log_params(params)
    return active_run


def end_run(status: str = "FINISHED") -> None:
    """End the currently active MLflow run started via `start_run()`."""
    mlflow.end_run(status=status)


@contextmanager
def mlflow_run(
    run_id: str, params: dict[str, Any] | None = None, settings: Settings | None = None
) -> Iterator[ActiveRun]:
    """Context-manager convenience wrapping `start_run`/`end_run`.

    Ends the run with status `FAILED` if the wrapped block raises, `FINISHED`
    otherwise -- mirrors `mlflow.start_run`'s own `with`-block semantics.
    """
    active_run = start_run(run_id, params=params, settings=settings)
    try:
        yield active_run
    except BaseException:
        end_run(status="FAILED")
        raise
    else:
        end_run(status="FINISHED")
