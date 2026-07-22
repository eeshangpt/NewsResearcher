from unittest.mock import patch

import pytest
from mlflow.tracking import MlflowClient
from testcontainers.postgres import PostgresContainer
from typer.testing import CliRunner

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.cli import app
from newsresearch.observability.mlflow_setup import EXPERIMENT_NAME
from newsresearch.persistence.db import init_db

# Hermetic `testcontainers[postgres]` per Story 0.4/0.7's own precedent, so
# this doesn't depend on the dev `docker compose up -d` stack being up.
# `LANGFUSE_HOST` points at an unreachable port -- the Langfuse callback
# handler enqueues traces asynchronously and never blocks/raises on a
# construction-time connectivity check, so this stays hermetic too; real
# trace delivery is verified manually against the live stack per Task 0.7.4.

runner = CliRunner()


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
def cli_env(tmp_path, monkeypatch, postgres_url):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEWSRESEARCH_DATABASE_URL", postgres_url)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-dummy")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:9")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    return postgres_url


def test_run_invokes_the_graph_end_to_end_and_exits_0(cli_env):
    result = runner.invoke(app, ["run", "test topic"])

    assert result.exit_code == 0
    assert "completed" in result.stdout


def test_run_writes_a_fully_populated_run_costs_row(cli_env):
    result = runner.invoke(app, ["run", "test topic"])
    assert result.exit_code == 0
    run_id = result.stdout.split("run_id=")[1].split(" ")[0]

    pool = init_db(cli_env)
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT stage, model, input_tokens, output_tokens FROM run_costs WHERE run_id = %s",
            (run_id,),
        ).fetchall()
    pool.close()

    assert len(rows) == 1
    stage, model, input_tokens, output_tokens = rows[0]
    assert stage == "subtopic"
    assert model == "stub-subtopic-model"
    assert input_tokens == 12
    assert output_tokens == 4


def test_run_produces_exactly_one_mlflow_run_tagged_with_run_id(cli_env):
    result = runner.invoke(app, ["run", "test topic"])
    assert result.exit_code == 0
    run_id = result.stdout.split("run_id=")[1].split(" ")[0]

    # `mlflow_run` (invoked inside the CLI command) already called
    # `mlflow.set_tracking_uri` against this test's scratch `mlruns` dir, so
    # a bare `MlflowClient()` picks that up via MLflow's own global state.
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    runs = client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.run_id = '{run_id}'"
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"


def test_run_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWSRESEARCH_DATABASE_URL", raising=False)

    result = runner.invoke(app, ["run", "test topic"])

    assert result.exit_code != 0


# Story 1.10 -- `dev sourcing-test` is a thin CLI wrapper over
# `agents/sourcing_agent.py` (already end-to-end verified against real
# GDELT/RSS by Story 1.9's own live test). Mocking `sourcing_agent` here
# keeps this hermetic while still exercising the CLI's own plumbing: arg
# parsing, keyword splitting, pool lifecycle, and output formatting.
def test_dev_sourcing_test_invokes_sourcing_agent_and_exits_0(cli_env):
    scored = ScoredArticle(
        article={"title": "Example", "url": "https://example.com/a", "domain": "example.com"},
        reputation_score=0.87,
        reputation_tier="major",
    )
    with patch("newsresearch.cli.sourcing_agent", return_value=[scored]) as mock_sourcing_agent:
        result = runner.invoke(app, ["dev", "sourcing-test", "climate policy"])

    assert result.exit_code == 0
    mock_sourcing_agent.assert_called_once()
    called_keywords, called_lookback_days = mock_sourcing_agent.call_args[0]
    assert called_keywords == ["climate", "policy"]
    assert called_lookback_days == 7
    assert "https://example.com/a" in result.stdout
    assert "example.com" in result.stdout
    assert "0.87" in result.stdout
    assert "major" in result.stdout


def test_dev_sourcing_test_prints_no_results_message_when_empty(cli_env):
    with patch("newsresearch.cli.sourcing_agent", return_value=[]):
        result = runner.invoke(app, ["dev", "sourcing-test", "quiet topic"])

    assert result.exit_code == 0
    assert "No articles" in result.stdout


def test_dev_sourcing_test_accepts_lookback_days_option(cli_env):
    with patch("newsresearch.cli.sourcing_agent", return_value=[]) as mock_sourcing_agent:
        result = runner.invoke(
            app, ["dev", "sourcing-test", "climate", "--lookback-days", "3"]
        )

    assert result.exit_code == 0
    _, called_lookback_days = mock_sourcing_agent.call_args[0]
    assert called_lookback_days == 3


def test_dev_sourcing_test_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWSRESEARCH_DATABASE_URL", raising=False)

    result = runner.invoke(app, ["dev", "sourcing-test", "climate policy"])

    assert result.exit_code != 0
