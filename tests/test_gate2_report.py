"""Gate 2 report aggregation (Task 2.6.1)."""

from unittest.mock import patch

from newsresearch.reports.gate2_report import build_gate2_report

CLUSTERING_RESULT = {
    "subtopic_id": "sub-1",
    "label": "Example Subtopic",
    "clusters": {
        0: [
            {"title": "Headline A", "domain": "a.com"},
            {"title": "Headline B", "domain": "b.com"},
            {"title": "Headline C", "domain": "a.com"},
        ],
        1: [
            {"title": "Headline D", "domain": "c.com"},
        ],
    },
    "noise": [
        {"title": "Headline E", "domain": "d.com"},
    ],
    "total_articles": 5,
}


def test_build_gate2_report_aggregates_expected_shape():
    report = build_gate2_report(CLUSTERING_RESULT)

    assert report["cluster_sizes"] == [3, 1]
    assert report["sample_headlines"] == ["Headline A", "Headline B", "Headline D"]
    assert report["source_spread"] == {"a.com": 2, "b.com": 1, "c.com": 1, "d.com": 1}


def test_build_gate2_report_never_calls_get_chat_model():
    with patch("newsresearch.llm.models.get_chat_model") as mock_get_chat_model:
        build_gate2_report(CLUSTERING_RESULT)
        mock_get_chat_model.assert_not_called()


def test_build_gate2_report_handles_empty_clusters_and_noise():
    report = build_gate2_report(
        {"subtopic_id": "sub-2", "label": "Empty", "clusters": {}, "noise": [], "total_articles": 0}
    )

    assert report == {"cluster_sizes": [], "sample_headlines": [], "source_spread": {}}
