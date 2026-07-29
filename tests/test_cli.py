import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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


# `cli_env`'s default subtopic-pipeline stubs produce an empty `candidates`
# list, so Gate 1's own render/prompt still fires (it interrupts
# unconditionally) but `fan_out` falls back to its no-candidates path --
# `fan_out_router` -- with no Gate 2 interrupts to resume. One "a\n" (approve)
# is enough stdin for these baseline plumbing tests.
def test_run_invokes_the_graph_end_to_end_and_exits_0(cli_env):
    result = runner.invoke(app, ["run", "test topic"], input="a\n")

    assert result.exit_code == 0, result.output
    assert "completed" in result.stdout


def test_run_writes_a_fully_populated_run_costs_row(cli_env):
    result = runner.invoke(app, ["run", "test topic"], input="a\n")
    assert result.exit_code == 0, result.output
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
    result = runner.invoke(app, ["run", "test topic"], input="a\n")
    assert result.exit_code == 0, result.output
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


# Task 2.7.1 -- `run`'s Gate 1/Gate 2 stdin harness (replaces PR #35's
# unconditional Gate 1 auto-approve stopgap). Mocking only external
# boundaries (`topical_clustering_agent`'s `sourcing_agent`/`embed`), same
# convention `test_graph_build.py`'s fan-out tests already established.
def test_run_approve_path_renders_gate1_then_each_gate2_report_and_completes(cli_env, monkeypatch):
    monkeypatch.setattr(
        "newsresearch.graph.build.rank_and_cap_subtopics",
        lambda *a, **kw: {
            "candidates": [
                {"label": "eu ai act", "article_count": 12},
                {"label": "us executive order", "article_count": 8},
            ],
            "excess": [{"label": "china ai regulation", "article_count": 3}],
        },
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    # Gate 1: approve ("a"). Two approved candidates fan out into two
    # independent Gate 2 interrupts, each resumed by a blank "continue".
    result = runner.invoke(app, ["run", "test topic"], input="a\n\n\n")

    assert result.exit_code == 0, result.output
    assert "Gate 1: Subtopic Candidates" in result.output
    assert "eu ai act" in result.output
    assert "us executive order" in result.output
    assert "china ai regulation" in result.output  # excess rendered too
    assert result.output.count("Gate 2: Cluster Report") == 2
    assert "completed" in result.output


def test_run_edit_path_reruns_real_reconciliation_with_a_visibly_different_candidate_set(
    cli_env, monkeypatch
):
    clustering = json.loads((FIXTURES_DIR / "clustering_synthetic_topics.json").read_text())
    merge_fixture = json.loads((FIXTURES_DIR / "reconciliation_merge.json").read_text())

    article_vectors = np.array(clustering["embeddings"])
    # Drop the last of 4 raw candidates via the CLI edit prompt; real
    # reconciliation then merges two of the surviving three near-duplicate
    # EU AI Act candidates into one, so exactly 2 land at Gate 2 -- proving
    # the edit path re-triggers real reconciliation, not an identity
    # pass-through of the edited list.
    candidate_vectors = np.array(merge_fixture["candidate_embeddings"][:-1])

    def fake_embed(texts):
        if len(texts) == len(article_vectors):
            return article_vectors
        if len(texts) == len(candidate_vectors):
            return candidate_vectors
        raise AssertionError(f"unexpected embed() call with {len(texts)} texts")

    monkeypatch.setattr("newsresearch.agents.subtopic_agent.embed", fake_embed)
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    articles = [{"title": s} for s in clustering["sentences"]]
    monkeypatch.setattr("newsresearch.graph.build.broad_topic_fetch", lambda *a, **kw: articles)
    monkeypatch.setattr(
        "newsresearch.graph.build.rank_and_cap_subtopics",
        lambda *a, **kw: {
            "candidates": [
                {"label": label, "article_count": 0} for label in merge_fixture["candidate_labels"]
            ],
            "excess": [],
        },
    )

    # Gate 1: edit ("e"), keep indices 0,1,2 -- drops the 4th raw candidate.
    # Gate 2: two branches (post-merge), each resumed by a blank "continue".
    result = runner.invoke(app, ["run", "test topic"], input="e\n0,1,2\n\n\n")

    assert result.exit_code == 0, result.output
    # Real reconciliation (not `stub_reconcile`'s identity pass-through): the
    # 3 kept-by-index candidates merge down to 2 distinct subtopics before
    # reaching Gate 2 -- an identity pass-through would have produced 3.
    assert result.output.count("Gate 2: Cluster Report") == 2


# Task 2.8.3 -- `--thread-id` lets a killed/interrupted `run` be rejoined
# instead of always minting a brand-new run (root cause of Task 2.8.2's
# durability walkthrough failing at the CLI layer, though the underlying
# `PostgresSaver` guarantee was already proven real by Task 2.3.2).
def test_run_prints_run_id_before_the_first_gate_so_a_kill_still_has_something_to_resume(cli_env):
    # Empty stdin starves Gate 1's prompt mid-render, aborting the process --
    # simulating a Ctrl-C kill while parked at the gate. The `run_id` must
    # already be on stdout by then (printed before `graph.invoke` is ever
    # called), not just at completion.
    result = runner.invoke(app, ["run", "test topic"], input="")

    assert result.exit_code != 0
    assert "run_id=run-" in result.output
    assert "--thread-id=run-" in result.output


def test_run_with_thread_id_resumes_a_killed_run_without_reseeding(cli_env):
    started = runner.invoke(app, ["run", "test topic"], input="")
    assert started.exit_code != 0
    run_id = started.output.split("run_id=")[1].split(" ")[0]

    resumed = runner.invoke(
        app, ["run", "ignored topic", "--thread-id", run_id], input="a\n"
    )

    assert resumed.exit_code == 0, resumed.output
    assert "(resuming)" in resumed.output
    assert "Gate 1: Subtopic Candidates" in resumed.output
    assert f"run_id={run_id} topic='test topic'" in resumed.output  # original topic, not reseeded
    assert "completed" in resumed.output


def test_run_with_unknown_thread_id_fails_readably_not_with_a_crash(cli_env):
    result = runner.invoke(app, ["run", "test topic", "--thread-id", "run-does-not-exist"])

    assert result.exit_code != 0
    assert "No pending interrupt" in result.output
    assert "Traceback" not in result.output


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
