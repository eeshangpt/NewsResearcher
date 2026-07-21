from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from testcontainers.postgres import PostgresContainer

from newsresearch.observability.cost_callback import CostCallbackHandler
from newsresearch.persistence.db import init_db

# Hermetic `testcontainers[postgres]`, per Story 0.4's own precedent -- no
# dependency on the dev `docker compose up -d` stack being up.


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
def pool(postgres_url):
    pool = init_db(postgres_url)
    yield pool
    pool.close()


@pytest.fixture
def pipeline_run_id(pool) -> str:
    """A fresh `runs` row per test, satisfying `run_costs.run_id`'s FK."""
    run_id = f"run-{uuid4()}"
    with pool.connection() as conn:
        conn.execute("INSERT INTO runs (run_id) VALUES (%s)", (run_id,))
    return run_id


def _chat_model_response(input_tokens=10, output_tokens=5) -> LLMResult:
    message = AIMessage(
        content="hello",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_on_llm_end_writes_a_fully_populated_run_costs_row(pool, pipeline_run_id):
    handler = CostCallbackHandler(pool)
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4.1-mini"}},
        messages=[[]],
        run_id=run_id,
        metadata={"run_id": pipeline_run_id, "stage": "subtopic"},
    )
    handler.on_llm_end(_chat_model_response(), run_id=run_id)

    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT run_id, stage, model, input_tokens, output_tokens, "
            "estimated_cost, latency_ms, created_at FROM run_costs WHERE run_id = %s",
            (pipeline_run_id,),
        ).fetchall()

    assert len(rows) == 1
    (
        row_run_id,
        stage,
        model,
        input_tokens,
        output_tokens,
        estimated_cost,
        latency_ms,
        created_at,
    ) = rows[0]
    assert row_run_id == pipeline_run_id
    assert stage == "subtopic"
    assert model == "gpt-4.1-mini"
    assert input_tokens == 10
    assert output_tokens == 5
    assert estimated_cost == pytest.approx((10 / 1000) * 0.0004 + (5 / 1000) * 0.0016)
    assert latency_ms is not None and latency_ms >= 0
    assert created_at is not None


def test_on_llm_error_drops_pending_state_without_writing_a_row(pool, pipeline_run_id):
    handler = CostCallbackHandler(pool)
    run_id = uuid4()

    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4.1-mini"}},
        messages=[[]],
        run_id=run_id,
        metadata={"run_id": pipeline_run_id, "stage": "subtopic"},
    )
    handler.on_llm_error(RuntimeError("boom"), run_id=run_id)

    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id FROM run_costs WHERE run_id = %s", (pipeline_run_id,)
        ).fetchall()
    assert rows == []


def test_db_write_failure_logs_and_continues_never_raises(pool, pipeline_run_id):
    handler = CostCallbackHandler(pool)
    pool.close()  # force the write to fail

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4.1-mini"}},
        messages=[[]],
        run_id=run_id,
        metadata={"run_id": pipeline_run_id, "stage": "subtopic"},
    )

    # Must not raise despite the pool being closed underneath it (NFR-3).
    handler.on_llm_end(_chat_model_response(), run_id=run_id)
