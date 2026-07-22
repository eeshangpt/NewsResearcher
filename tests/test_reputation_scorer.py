import pytest

from newsresearch.config import Settings
from newsresearch.reputation import scorer

SETTINGS = Settings(
    reputation={
        "base_score_wire": 0.8,
        "base_score_major": 0.7,
        "base_score_unknown": 0.3,
        "weight_domain_age": 0.10,
        "weight_backlink_proxy": 0.10,
        "weight_presence_frequency": 0.05,
        "weight_legitimacy_flags": 0.05,
        "adjustment_bound": 0.3,
    }
)


# --- get_base_tier_score -----------------------------------------------------


def test_get_base_tier_score_wire_domain():
    score, tier = scorer.get_base_tier_score("reuters.com", settings=SETTINGS)
    assert score == 0.8
    assert tier == "wire"


def test_get_base_tier_score_major_domain():
    score, tier = scorer.get_base_tier_score("bbc.com", settings=SETTINGS)
    assert score == 0.7
    assert tier == "major"


def test_get_base_tier_score_unknown_domain():
    score, tier = scorer.get_base_tier_score("some-random-blog.example", settings=SETTINGS)
    assert score == 0.3
    assert tier == "unknown"


def test_get_base_tier_score_is_case_and_prefix_insensitive_string_match():
    # Only exact case-normalized matches -- no www-stripping magic in the
    # scorer itself (that's the sourcing clients' job per the Task 1.4.2/1.9.1
    # integration-checkpoint note), just lowercase-normalization.
    score, tier = scorer.get_base_tier_score("REUTERS.com", settings=SETTINGS)
    assert score == 0.8
    assert tier == "wire"


# --- score_domain: exact formula output with fixed inputs + known weights ---


def test_score_domain_exact_formula_output_for_known_wire_domain():
    result = scorer.score_domain(
        domain="reuters.com",
        domain_age_years=10.0,  # norm = 10/20 = 0.5
        backlink_proxy=0.8,
        presence_frequency=1.0,
        https_present=True,
        about_page_present=True,
        settings=SETTINGS,
    )

    # legitimacy = (1.0 + 1.0) / 2 = 1.0
    expected_adjustment = 0.10 * 0.5 + 0.10 * 0.8 + 0.05 * 1.0 + 0.05 * 1.0
    assert expected_adjustment == pytest.approx(0.23)
    assert result == {
        "domain": "reuters.com",
        "tier": "wire",
        "base_score": 0.8,
        "heuristic_adjustment": pytest.approx(0.23),
        "final_score": pytest.approx(0.8 + 0.23),
    }


def test_score_domain_exact_formula_output_for_unknown_domain_with_neutral_signals():
    result = scorer.score_domain(
        domain="obscure-outlet.example",
        domain_age_years=None,  # neutral 0.5
        backlink_proxy=0.5,
        presence_frequency=None,  # neutral 0.5
        https_present=None,  # neutral 0.5
        about_page_present=None,  # neutral 0.5
        settings=SETTINGS,
    )

    # Every signal neutral (0.5) -> adjustment = 0.5 * sum(weights) = 0.5 * 0.3 = 0.15
    expected_adjustment = 0.5 * (0.10 + 0.10 + 0.05 + 0.05)
    assert expected_adjustment == pytest.approx(0.15)
    assert result == {
        "domain": "obscure-outlet.example",
        "tier": "unknown",
        "base_score": 0.3,
        "heuristic_adjustment": pytest.approx(0.15),
        "final_score": pytest.approx(0.3 + 0.15),
    }


def test_score_domain_exact_formula_output_for_major_domain_all_signals_absent_or_negative():
    result = scorer.score_domain(
        domain="bbc.com",
        domain_age_years=0.0,
        backlink_proxy=0.0,
        presence_frequency=0.0,
        https_present=False,
        about_page_present=False,
        settings=SETTINGS,
    )

    assert result == {
        "domain": "bbc.com",
        "tier": "major",
        "base_score": 0.7,
        "heuristic_adjustment": pytest.approx(0.0),
        "final_score": pytest.approx(0.7),
    }


# --- score_domain: adversarial extreme-signal combination is clipped -------


def test_score_domain_clips_adjustment_when_weights_pushed_far_beyond_bound():
    adversarial_settings = Settings(
        reputation={
            **SETTINGS.reputation.model_dump(),
            # Deliberately misconfigured weights whose sum-at-max (20.0) is
            # far beyond adjustment_bound (0.3) -- simulates either a bad
            # config override or a signal drifting outside [0, 1].
            "weight_domain_age": 5.0,
            "weight_backlink_proxy": 5.0,
            "weight_presence_frequency": 5.0,
            "weight_legitimacy_flags": 5.0,
            "adjustment_bound": 0.3,
        }
    )

    result = scorer.score_domain(
        domain="reuters.com",
        domain_age_years=1_000_000.0,  # would normalize far above 1.0 unclipped
        backlink_proxy=1.0,
        presence_frequency=1.0,
        https_present=True,
        about_page_present=True,
        settings=adversarial_settings,
    )

    # Confirmed clipped to the configured bound, not the (20.0) raw sum.
    assert result["heuristic_adjustment"] == pytest.approx(0.3)
    assert result["final_score"] == pytest.approx(0.8 + 0.3)
    assert result["final_score"] < 0.8 + 20.0


def test_score_domain_clips_negative_direction_too():
    adversarial_settings = Settings(
        reputation={
            **SETTINGS.reputation.model_dump(),
            "weight_domain_age": -5.0,
            "weight_backlink_proxy": -5.0,
            "weight_presence_frequency": -5.0,
            "weight_legitimacy_flags": -5.0,
            "adjustment_bound": 0.3,
        }
    )

    result = scorer.score_domain(
        domain="bbc.com",
        domain_age_years=100.0,
        backlink_proxy=1.0,
        presence_frequency=1.0,
        https_present=True,
        about_page_present=True,
        settings=adversarial_settings,
    )

    assert result["heuristic_adjustment"] == pytest.approx(-0.3)
    assert result["final_score"] == pytest.approx(0.7 - 0.3)


def test_score_domain_default_settings_used_when_none_provided():
    result = scorer.score_domain(
        domain="apnews.com",
        domain_age_years=20.0,
        backlink_proxy=1.0,
        presence_frequency=1.0,
        https_present=True,
        about_page_present=True,
    )

    default_settings = Settings()
    assert result["base_score"] == default_settings.reputation.base_score_wire
    assert result["heuristic_adjustment"] == pytest.approx(default_settings.reputation.adjustment_bound)
