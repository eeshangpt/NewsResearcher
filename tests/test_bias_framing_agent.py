"""Unit tests for `agents/bias_framing_agent.py` + its schema (Task 4.1.1b).

Mocks `get_chat_model`, `get_langfuse_callback_handler`, and the persistence
reads (no real DB, no real API key needed) -- matches
`tests/test_summarization_agent.py`'s established pattern. Label *quality* is
Task 4.1.1a's real-data evaluation, not asserted here; these prove the wiring
(prompt input format incl. the load-bearing distinct-article roster, batching,
additive merge, model stage, config forwarding).

The schema-property tests at the bottom are ported from
`notebooks/phase4_bias_framing_schema_draft.py`'s runnable asserts, so a later
well-meaning `bias_score: float` fails a check rather than passing review.
"""

import typing
from typing import Literal
from unittest.mock import MagicMock, patch

import pydantic
import pytest
from pydantic import BaseModel

from newsresearch.agents.bias_framing_agent import analyze_subtopic_framing
from newsresearch.config import Settings
from newsresearch.llm.schemas import ArticleFraming, BiasFramingBatch, ClusterFraming

# Two articles from ONE domain plus a second domain, and one omitting row:
# the "2 articles, 1 domain" shape is exactly what the roster line exists to
# stop the model miscounting (Task 4.1.1a iteration 4).
MOCK_ROWS = {
    "sub1:0": [
        {
            "article_id": "a1",
            "relation": "asserts",
            "claim_text": "Police removed the detonator.",
            "sentiment_label": "neutral",
            "sentiment_compound": 0.0,
            "domain": "theguardian.com",
        },
        {
            "article_id": "a1",
            "relation": "asserts",
            "claim_text": "The device was made safe on site.",
            "sentiment_label": "neutral",
            "sentiment_compound": 0.0,
            "domain": "theguardian.com",
        },
        {
            "article_id": "a2",
            "relation": "asserts",
            "claim_text": "The bomb did not explode because the detonator was faulty.",
            "sentiment_label": "neutral",
            "sentiment_compound": 0.0,
            "domain": "theguardian.com",
        },
        {
            "article_id": "a3",
            "relation": "omits",
            "claim_text": None,
            "sentiment_label": None,
            "sentiment_compound": None,
            "domain": "bbc.com",
        },
    ],
    "sub1:1": [
        {
            "article_id": "a3",
            "relation": "asserts",
            "claim_text": "The ambassador blamed Moscow.",
            "sentiment_label": "neutral",
            "sentiment_compound": 0.0,
            "domain": "bbc.com",
        }
    ],
    "sub1:2": [
        {
            "article_id": "a1",
            "relation": "asserts",
            "claim_text": "The airport is a NATO logistics hub.",
            "sentiment_label": "neutral",
            "sentiment_compound": 0.0,
            "domain": "theguardian.com",
        }
    ],
    # Only omitting rows -- nothing for the model to describe.
    "sub1:3": [
        {
            "article_id": "a2",
            "relation": "omits",
            "claim_text": None,
            "sentiment_label": None,
            "sentiment_compound": None,
            "domain": "theguardian.com",
        }
    ],
}


def _framing(cluster_id: str) -> ClusterFraming:
    return ClusterFraming(
        cluster_id=cluster_id,
        cluster_coherence="single_shared_fact",
        shared_focus="what the cluster is about",
        divergence="no_notable_divergence",
        divergence_note="a1 and a2 word it similarly.",
        per_article=[
            ArticleFraming(
                article_id="a1",
                domain="theguardian.com",
                epistemic_treatment="states_as_established",
                contextual_frame="keeps to procedural police response",
                evidence_quote="removed the detonator",
            )
        ],
    )


class _RecordingChatModel:
    """See `tests/test_summarization_agent.py`'s identical helper: an explicit
    `config` parameter makes `RunnableLambda` actually forward the run config
    through, unlike a bare `MagicMock`.

    Echoes back one `ClusterFraming` per `Cluster <id>` block it was actually
    given, so batching and the additive merge are measured against the real
    rendered prompt rather than a canned response.
    """

    def __init__(self):
        self.calls: list[tuple[object, dict | None]] = []
        self.structured_output_schema = None

    def with_structured_output(self, schema):
        self.structured_output_schema = schema
        return self

    def __call__(self, prompt_value, config=None):
        self.calls.append((prompt_value, config))
        rendered = prompt_value.to_string()
        ids = [
            line.split(" ", 1)[1].strip()
            for line in rendered.splitlines()
            if line.startswith("Cluster ")
        ]
        return BiasFramingBatch(clusters=[_framing(cid) for cid in ids])

    @property
    def call_args(self):
        prompt_value, config = self.calls[-1]
        return (prompt_value,), {"config": config}

    def rendered(self, i: int = -1) -> str:
        return self.calls[i][0].to_string()


def _patched(fn):
    """The four patches every wiring test needs, in the argument order below.

    First patch applied binds the first argument, so this list is in the same
    order as the test signatures that consume it.
    """
    fn = patch("newsresearch.agents.bias_framing_agent.get_chat_model")(fn)
    fn = patch("newsresearch.agents.bias_framing_agent.get_langfuse_callback_handler")(fn)
    fn = patch("newsresearch.agents.bias_framing_agent.read_subtopic_cluster_ids")(fn)
    fn = patch("newsresearch.agents.bias_framing_agent.read_cluster_article_relations")(fn)
    return fn


def _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows):
    model = _RecordingChatModel()
    mock_get_chat_model.return_value = model
    mock_get_langfuse.return_value = MagicMock()
    mock_read_ids.return_value = list(MOCK_ROWS)
    mock_read_rows.side_effect = lambda pool, cluster_id: MOCK_ROWS[cluster_id]
    return model


@_patched
def test_uses_bias_framing_stage_with_structured_output(
    mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows
):
    model = _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows)

    result = analyze_subtopic_framing(MagicMock(), "sub1", run_id="run-1")

    mock_get_chat_model.assert_called_once_with("bias_framing")
    assert model.structured_output_schema is BiasFramingBatch
    # sub1:3 holds only omitting rows and is dropped, so 3 clusters remain.
    assert [c.cluster_id for c in result] == ["sub1:0", "sub1:1", "sub1:2"]


@_patched
def test_batches_by_batch_size_and_merges_additively(
    mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows
):
    model = _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows)

    result = analyze_subtopic_framing(
        MagicMock(), "sub1", settings=Settings(models={"bias_framing_batch_size": 2})
    )

    assert len(model.calls) == 2
    assert "Cluster sub1:0" in model.rendered(0) and "Cluster sub1:1" in model.rendered(0)
    assert "Cluster sub1:2" not in model.rendered(0)
    assert "Cluster sub1:2" in model.rendered(1)
    # Every batch's clusters merged into one list, nothing dropped or overwritten.
    assert [c.cluster_id for c in result] == ["sub1:0", "sub1:1", "sub1:2"]


@_patched
def test_prompt_carries_distinct_article_roster_and_only_asserting_rows(
    mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows
):
    model = _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows)

    analyze_subtopic_framing(MagicMock(), "sub1")
    rendered = model.rendered()

    # The roster is load-bearing: without it the model reads "one domain with
    # two articles" as one article (Task 4.1.1a, 0/69 -> 6/69 regression).
    # a1 contributes two claim lines but must appear once, and the count is 2,
    # not 3 lines and not 1 domain.
    assert "Distinct articles in this cluster (2): a1, a2" in rendered
    assert "Distinct articles in this cluster (1): a3" in rendered
    assert "Claims (article id | source domain | claim):" in rendered
    assert "- a1 | theguardian.com | Police removed the detonator." in rendered
    # Omitting rows are not claims and never reach the model.
    assert "a3 | bbc.com | None" not in rendered
    assert "Cluster sub1:3" not in rendered
    # Real prompt file content, not a stub template.
    assert "You are the Bias & Framing Agent" in rendered


@_patched
def test_attaches_langfuse_and_merges_ambient_config(
    mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows
):
    model = _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows)
    mock_handler = MagicMock()
    mock_get_langfuse.return_value = mock_handler

    ambient_callback = MagicMock()
    analyze_subtopic_framing(
        MagicMock(), "sub1", run_id="run-3", config={"callbacks": [ambient_callback]}
    )

    _, call_kwargs = model.call_args
    config = call_kwargs["config"]
    assert mock_handler in config["callbacks"].handlers
    assert ambient_callback in config["callbacks"].handlers
    assert config["metadata"]["stage"] == "bias_framing"
    assert config["metadata"]["run_id"] == "run-3"
    assert config["metadata"]["subtopic_id"] == "sub1"


@_patched
def test_no_clusters_makes_no_llm_call(
    mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows
):
    model = _wire(mock_get_chat_model, mock_get_langfuse, mock_read_ids, mock_read_rows)
    mock_read_ids.return_value = []

    assert analyze_subtopic_framing(MagicMock(), "sub1") == []
    assert model.calls == []


# --- Schema properties (ported from Task 4.1.1a's runnable asserts) ---------

BANNED = {
    "left",
    "right",
    "center",
    "centre",
    "lean",
    "leaning",
    "liberal",
    "conservative",
    "progressive",
    "partisan",
    "ideology",
    "ideological",
    "bias_score",
    "political",
}


def test_schema_has_no_numeric_or_political_field_anywhere():
    """Fails if anyone adds a numeric field or a political-lean term."""

    def walk(model: type[BaseModel], seen: set[type[BaseModel]]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            ann = field.annotation
            assert not any(b in name.lower() for b in BANNED), f"political field name: {name}"
            # A left/center/right scale's natural encoding is a number, so the
            # schema deliberately offers no numeric slot at any nesting level.
            assert ann not in (int, float), f"numeric field in framing schema: {name}"
            for member in typing.get_args(ann) if typing.get_origin(ann) is Literal else ():
                assert isinstance(member, str)
                assert not any(b in member.lower() for b in BANNED), f"political enum value: {member}"
            for candidate in (ann, *typing.get_args(ann)):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    walk(candidate, seen)

    walk(BiasFramingBatch, set())


@pytest.mark.parametrize("smuggled", ["left_leaning", "centre", "conservative", "3"])
def test_schema_enums_reject_smuggled_lean_labels(smuggled):
    """`Literal` really rejects a lean label, it does not merely discourage it."""
    with pytest.raises(pydantic.ValidationError):
        ArticleFraming(
            article_id="a1",
            domain="example.com",
            epistemic_treatment=smuggled,
            contextual_frame="x",
            evidence_quote="y",
        )


def test_schema_accepts_a_valid_epistemic_treatment():
    ok = ArticleFraming(
        article_id="a1",
        domain="example.com",
        epistemic_treatment="hedges_as_uncertain",
        contextual_frame="plain factual statement, no wider framing",
        evidence_quote="under ongoing investigation",
    )
    assert ok.epistemic_treatment == "hedges_as_uncertain"
