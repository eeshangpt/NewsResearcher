import math
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from newsresearch.reputation import signals


# --- Task 1.6.1: WHOIS domain-age signal ------------------------------------


def test_get_domain_age_years_returns_none_when_whois_raises(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("whois server refused connection")

    monkeypatch.setattr(signals.whois, "whois", _raise)

    assert signals.get_domain_age_years("example.com") is None


def test_get_domain_age_years_returns_none_on_timeout(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("WHOIS lookup timed out")

    monkeypatch.setattr(signals.whois, "whois", _raise_timeout)

    assert signals.get_domain_age_years("example.com") is None


def test_get_domain_age_years_computes_plausible_age_from_creation_date(monkeypatch):
    creation_date = datetime.now(timezone.utc) - timedelta(days=3652)  # ~10 years

    monkeypatch.setattr(
        signals.whois, "whois", lambda *a, **k: {"creation_date": creation_date}
    )

    age = signals.get_domain_age_years("example.com")

    assert age == pytest.approx(10.0, abs=0.05)


# --- Task 1.6.2: Tranco backlink/popularity-proxy signal --------------------


@pytest.fixture(autouse=True)
def _reset_tranco_cache():
    signals._tranco_ranks_cache = None
    yield
    signals._tranco_ranks_cache = None


def test_get_backlink_proxy_score_known_high_rank_domain_is_near_one():
    # google.com is rank 1 in the committed Tranco top-100k snapshot.
    score = signals.get_backlink_proxy_score("google.com")

    assert score == pytest.approx(1.0)


def test_get_backlink_proxy_score_domain_absent_from_snapshot_is_neutral():
    score = signals.get_backlink_proxy_score("definitely-not-a-real-outlet-domain-xyz123.test")

    assert score == 0.5


def test_get_backlink_proxy_score_matches_log_scale_formula_for_fixed_rank(tmp_path, monkeypatch):
    csv_path = tmp_path / "tranco_fixture.csv"
    csv_path.write_text("1000,midrank.example\n")
    monkeypatch.setattr(signals, "_TRANCO_CSV_PATH", csv_path)
    signals._tranco_ranks_cache = None

    score = signals.get_backlink_proxy_score("midrank.example")

    expected = 1 - math.log10(1000) / math.log10(100_000)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.4)


# --- Task 1.6.3: HTTPS + about-page legitimacy heuristic --------------------


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_check_https_and_about_page_detects_both_flags(monkeypatch):
    def _fake_head(url, timeout=None, follow_redirects=None):
        return _FakeResponse(200)

    monkeypatch.setattr(signals.httpx, "head", _fake_head)

    result = signals.check_https_and_about_page("example.com")

    assert result == {"https_present": True, "about_page_present": True}


def test_check_https_and_about_page_no_about_page_found(monkeypatch):
    def _fake_head(url, timeout=None, follow_redirects=None):
        if url.rstrip("/").endswith("example.com"):
            return _FakeResponse(200)
        return _FakeResponse(404)

    monkeypatch.setattr(signals.httpx, "head", _fake_head)

    result = signals.check_https_and_about_page("example.com")

    assert result == {"https_present": True, "about_page_present": False}


def test_check_https_and_about_page_fails_soft_on_timeout(monkeypatch):
    def _raise_timeout(url, timeout=None, follow_redirects=None):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(signals.httpx, "head", _raise_timeout)

    result = signals.check_https_and_about_page("unreachable-domain.test")

    assert result == {"https_present": None, "about_page_present": None}
