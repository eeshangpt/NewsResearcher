"""Application configuration (NFR-5).

One `Settings` object loaded from `.env` (secrets/connection strings) plus an
optional `config.yaml` (tunables). Env vars always take precedence over the
yaml file. Every agent/module should read tunables from here rather than
hardcoding a constant that's listed as configurable.
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class PipelineSettings(BaseModel):
    # TRD/PRD default: 5 subtopics carried forward past Gate 1.
    max_subtopics: int = 5


class ReputationSettings(BaseModel):
    # TRD 4.2: recompute cached domain reputation after this many days.
    staleness_days: int = 30
    min_score_threshold: float = 0.5


class ClusteringSettings(BaseModel):
    similarity_threshold: float = 0.75
    subtopic_match_threshold: float = 0.8


class ModelSettings(BaseModel):
    # Placeholder model names per stage (TRD 2: small model for mechanical
    # tasks, large model for framing/briefing) — finalized choices are
    # data-scientist's call, this only stubs the configurable slot.
    subtopic: str = "gpt-4.1"
    claim_extraction: str = "gpt-4.1-mini"
    summarization: str = "gpt-4.1-mini"
    bias_framing: str = "gpt-4.1"
    briefing: str = "gpt-4.1"


class EmbeddingsSettings(BaseModel):
    backend: Literal["local", "openai"] = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    database_url: str | None = Field(default=None, validation_alias="NEWSRESEARCH_DATABASE_URL")
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", validation_alias="LANGFUSE_HOST")
    mlflow_tracking_uri: str = Field(default="./mlruns", validation_alias="MLFLOW_TRACKING_URI")

    pipeline: PipelineSettings = PipelineSettings()
    reputation: ReputationSettings = ReputationSettings()
    clustering: ClusteringSettings = ClusteringSettings()
    models: ModelSettings = ModelSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (highest first): explicit init kwargs, real env vars,
        # .env file, config.yaml tunables, secrets file. Yaml sits below env
        # so env vars always override yaml, per NFR-5.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
