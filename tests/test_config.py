import shutil
from pathlib import Path

import pytest

from newsresearch.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run Settings() against a scratch directory with no ambient .env/config.yaml."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "OPENAI_API_KEY",
        "NEWSRESEARCH_DATABASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "MLFLOW_TRACKING_URI",
        "PIPELINE__MAX_SUBTOPICS",
        "SOURCING__MIN_PRIMARY_ARTICLE_COUNT",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_settings_instantiates_with_defaults_and_every_nested_field_present(isolated_cwd):
    settings = Settings()

    assert isinstance(settings.pipeline.max_subtopics, int)
    assert isinstance(settings.reputation.staleness_days, int)
    assert isinstance(settings.reputation.min_score_threshold, float)
    assert isinstance(settings.clustering.similarity_threshold, float)
    assert isinstance(settings.clustering.subtopic_match_threshold, float)
    assert isinstance(settings.sourcing.min_primary_article_count, int)
    assert isinstance(settings.models.subtopic, str)
    assert isinstance(settings.models.claim_extraction, str)
    assert isinstance(settings.models.summarization, str)
    assert isinstance(settings.models.bias_framing, str)
    assert isinstance(settings.models.briefing, str)
    assert settings.embeddings.backend in ("local", "openai")
    assert settings.embeddings.backend == "local"


def test_env_example_shaped_env_loads_without_missing_required_field_error(isolated_cwd):
    env_file = isolated_cwd / ".env"
    shutil.copy(ENV_EXAMPLE, env_file)

    settings = Settings()

    assert settings.database_url == "postgresql://newsresearch:newsresearch@localhost:5432/newsresearch"
    assert settings.langfuse_host == "http://localhost:3000"
    assert settings.mlflow_tracking_uri == "./mlruns"
    assert settings.openai_api_key in (None, "")
    assert settings.langfuse_public_key in (None, "")


def test_config_yaml_supplies_tunables(isolated_cwd):
    (isolated_cwd / "config.yaml").write_text(
        "pipeline:\n"
        "  max_subtopics: 7\n"
        "reputation:\n"
        "  staleness_days: 45\n"
        "embeddings:\n"
        "  backend: openai\n"
    )

    settings = Settings()

    assert settings.pipeline.max_subtopics == 7
    assert settings.reputation.staleness_days == 45
    assert settings.embeddings.backend == "openai"


def test_env_vars_override_yaml_tunables(isolated_cwd, monkeypatch):
    (isolated_cwd / "config.yaml").write_text("pipeline:\n  max_subtopics: 7\n")
    monkeypatch.setenv("PIPELINE__MAX_SUBTOPICS", "99")

    settings = Settings()

    assert settings.pipeline.max_subtopics == 99


def test_sourcing_min_primary_article_count_defaults_and_overrides(isolated_cwd, monkeypatch):
    default_settings = Settings()
    assert default_settings.sourcing.min_primary_article_count == 15

    (isolated_cwd / "config.yaml").write_text("sourcing:\n  min_primary_article_count: 20\n")
    assert Settings().sourcing.min_primary_article_count == 20

    monkeypatch.setenv("SOURCING__MIN_PRIMARY_ARTICLE_COUNT", "5")
    assert Settings().sourcing.min_primary_article_count == 5
