import pytest
from mlflow.tracking import MlflowClient

from newsresearch.config import Settings
from newsresearch.observability.mlflow_setup import EXPERIMENT_NAME, mlflow_run


@pytest.fixture
def file_store_settings(tmp_path, monkeypatch):
    """Isolate MLflow's file-store to a scratch dir, per test, so this test
    never writes into the repo's own `./mlruns`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    return Settings()


def _runs_tagged(run_id: str) -> list:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.run_id = '{run_id}'"
    )


def test_mlflow_run_produces_exactly_one_run_tagged_with_run_id(file_store_settings):
    run_id = "test-run-1"

    with mlflow_run(run_id, params={"max_subtopics": 5}, settings=file_store_settings):
        pass

    runs = _runs_tagged(run_id)
    assert len(runs) == 1
    assert runs[0].data.tags["run_id"] == run_id
    assert runs[0].data.params["max_subtopics"] == "5"
    assert runs[0].info.status == "FINISHED"


def test_mlflow_run_marks_failed_status_on_exception(file_store_settings):
    run_id = "test-run-2"

    with pytest.raises(ValueError):
        with mlflow_run(run_id, settings=file_store_settings):
            raise ValueError("boom")

    runs = _runs_tagged(run_id)
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
