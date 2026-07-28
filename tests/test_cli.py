from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from mlflow.tracking import MlflowClient
from testcontainers.postgres import PostgresContainer
from typer.testing import CliRunner

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.cli import app
from newsresearch.llm.schemas import SubtopicCandidateList
from newsresearch.observability.mlflow_setup import EXPERIMENT_NAME
from newsresearch.persistence.db import init_db
from newsresearch.sourcing.gdelt import GDELTError

# Hermetic `testcontainers[postgres]` per Story 0.4/0.7's own precedent, so
# this doesn't depend on the dev `docker compose up -d` stack being up.
# `LANGFUSE_HOST` points at an unreachable port -- the Langfuse callback
# handler enqueues traces asynchronously and never blocks/raises on a
# construction-time connectivity check, so this stays hermetic too; real
# trace delivery is verified manually against the live stack per Task 0.7.4.

runner = CliRunner()


class _FakeSubtopicChatModel(BaseChatModel):
    """Real `BaseChatModel` stand-in for `run`'s now-real `subtopic` node
    (Story 2.2 production wiring) -- fires genuine `on_chat_model_start`/
    `on_llm_end` callbacks (same usage-metadata shape/values Task 0.7.4's
    original stub used) so `test_run_writes_a_fully_populated_run_costs_row`
    keeps proving the cost-callback plumbing end-to-end, while
    `with_structured_output` short-circuits to a fixed candidate list so
    `propose_candidates` doesn't need a real `OPENAI_API_KEY`.
    """

    model_name: str = "stub-subtopic-model"

    @property
    def _llm_type(self) -> str:
        return "stub-subtopic-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            content="acknowledged",
            usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model_name": self.model_name},
        )

    def with_structured_output(self, schema, **kwargs):
        return self | RunnableLambda(lambda _: SubtopicCandidateList(candidates=[]))


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

    # `run`'s `subtopic` node is now real (Story 2.2 production wiring): mock
    # its LLM/sourcing/reconciliation calls so the CLI e2e tests stay
    # hermetic (no OpenAI key, no live GDELT/RSS), same convention
    # `test_graph_build.py`'s `_stub_subtopic_pipeline` already established.
    monkeypatch.setattr(
        "newsresearch.agents.subtopic_agent.get_chat_model", lambda stage: _FakeSubtopicChatModel()
    )
    monkeypatch.setattr("newsresearch.graph.build.broad_topic_fetch", lambda *a, **kw: [])
    monkeypatch.setattr(
        "newsresearch.graph.build.reconcile_subtopics",
        lambda *a, **kw: {"reconciled": [], "total_articles": 0},
    )
    monkeypatch.setattr(
        "newsresearch.graph.build.rank_and_cap_subtopics",
        lambda *a, **kw: {"candidates": [], "excess": []},
    )
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


# Tech-lead review follow-up: a `GDELTError` (GDELT is a primary source,
# allowed to hard-fail per NFR-3 -- only Google News backfill soft-fails)
# used to crash this command with a raw traceback. It should now surface as
# a readable diagnostic + non-zero exit instead.
def test_dev_sourcing_test_reports_a_gdelt_error_readably_instead_of_crashing(cli_env):
    with patch("newsresearch.cli.sourcing_agent", side_effect=GDELTError("boom")):
        result = runner.invoke(app, ["dev", "sourcing-test", "climate policy"])

    assert result.exit_code != 0
    assert "GDELT error" in result.output
    assert "Traceback" not in result.output
