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

    # TRD 4.2 `base_tier_score`: trusted-tier example is 0.7, unknown is 0.3.
    # `data/trusted_outlets.yaml` actually carries two trusted sub-tiers
    # (`wire`/`major`) rather than TRD's single flattened "trusted" bucket --
    # `wire` (global wire services with no editorial slant of their own:
    # Reuters, AP, AFP, UPI) is scored a notch above `major` outlets, both
    # above TRD's literal 0.7 example, since it's the strictly-higher-trust
    # tier per trusted_outlets.yaml's own docstring; `major` keeps TRD's
    # literal 0.7; `unknown` keeps TRD's literal 0.3 floor.
    base_score_wire: float = 0.8
    base_score_major: float = 0.7
    base_score_unknown: float = 0.3

    # TRD 4.2 `w1..w4`: "tunable weights summing to a bounded adjustment
    # range (e.g. ±0.3)". Defaults below sum to exactly `adjustment_bound`
    # when every normalized signal is at its max (1.0), matching that stated
    # intent; `adjustment_bound` is *also* enforced as an explicit clip in
    # `reputation/scorer.py` (not just relied on via weight configuration),
    # so a weight override or an out-of-range signal can never push a score
    # outside the configured bound.
    weight_domain_age: float = 0.10
    weight_backlink_proxy: float = 0.10
    weight_presence_frequency: float = 0.05
    weight_legitimacy_flags: float = 0.05
    adjustment_bound: float = 0.3


class ClusteringSettings(BaseModel):
    similarity_threshold: float = 0.75
    subtopic_match_threshold: float = 0.8


class SourcingSettings(BaseModel):
    # Below this many combined GDELT+RSS primary results, the sourcing
    # orchestrator falls back to Google News RSS backfill (Story 1.8). 15
    # is a sane middle ground: comfortably more than a single trusted-outlet
    # feed's typical per-topic yield (so a quiet news day still triggers
    # backfill) but high enough that a normal multi-source haul doesn't
    # trigger the unofficial-endpoint fallback needlessly (NFR-3).
    min_primary_article_count: int = 15


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
    sourcing: SourcingSettings = SourcingSettings()
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
