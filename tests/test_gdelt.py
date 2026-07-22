"""Unit tests for `sourcing/gdelt.py` (Story 1.2).

HTTP layer mocked via `respx` per this project's testing conventions
(`tests/live/` holds the opt-in real-API smoke test instead). The 250-record
capped fixture is a real GDELT DOC 2.0 response, captured once while
building this module (`tests/fixtures/gdelt_doc2_capped_250_climate.json`).
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx

from newsresearch.sourcing.gdelt import (
    GDELT_DOC_API_URL,
    GDELT_MAX_RECORDS_PER_CALL,
    GDELTError,
    _build_query,
    fetch,
    query_range,
    query_window,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CAPPED_250_FIXTURE = json.loads(
    (FIXTURES_DIR / "gdelt_doc2_capped_250_climate.json").read_text()
)


def _articles_payload(count: int, *, domain: str = "example.com") -> dict:
    return {
        "articles": [
            {
                "url": f"https://{domain}/story-{i}",
                "url_mobile": "",
                "title": f"Story {i}",
                "seendate": "20260722T101500Z",
                "socialimage": "",
                "domain": domain,
                "language": "English",
                "sourcecountry": "United States",
            }
            for i in range(count)
        ]
    }


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Backoff/inter-request delays would otherwise slow every test down."""
    monkeypatch.setattr("newsresearch.sourcing.gdelt.time.sleep", lambda _seconds: None)


def test_build_query_joins_and_quotes_multi_word_keywords_wrapped_in_parens():
    # Verified against the real GDELT endpoint (tech-lead review, Task
    # 1.10.1 follow-up): an OR'd multi-term query is rejected with HTTP 200
    # + "Queries containing OR'd terms must be surrounded by ()" unless the
    # whole expression is parenthesized.
    assert _build_query(["climate change", "wildfire"]) == '("climate change" OR wildfire)'


def test_build_query_does_not_quote_a_single_bare_token():
    # Verified against the real GDELT endpoint: quoting a single-token
    # keyword gets HTTP 200 + "The specified phrase is too short" --
    # phrase-quoting is for multi-word phrases only.
    assert _build_query(["Iran"]) == "Iran"


def test_build_query_quotes_a_single_multi_word_phrase_without_parens():
    assert _build_query(["climate change"]) == '"climate change"'


def test_build_query_rejects_empty_keyword_list():
    with pytest.raises(ValueError):
        _build_query([])


@respx.mock
def test_query_window_parses_real_captured_fixture_shape():
    respx.get(GDELT_DOC_API_URL).mock(
        return_value=httpx.Response(200, json=CAPPED_250_FIXTURE)
    )

    articles = query_window(
        '"climate"', datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 7, 22, tzinfo=timezone.utc)
    )

    assert len(articles) == 250
    first = articles[0]
    assert first["title"] == CAPPED_250_FIXTURE["articles"][0]["title"]
    assert first["url"] == CAPPED_250_FIXTURE["articles"][0]["url"]
    assert first["domain"] == "mikrometoxos.gr"
    assert first["published_at"] == datetime(2026, 7, 22, 12, 15, tzinfo=timezone.utc)
    assert first["source_type"] == "gdelt"


@respx.mock
def test_query_window_returns_empty_list_for_no_articles_key():
    respx.get(GDELT_DOC_API_URL).mock(return_value=httpx.Response(200, json={"articles": []}))

    articles = query_window(
        '"no such topic"',
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 7, 22, tzinfo=timezone.utc),
    )

    assert articles == []


def test_query_window_rejects_max_records_above_gdelt_cap():
    with pytest.raises(ValueError, match="250"):
        query_window(
            '"x"',
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 22, tzinfo=timezone.utc),
            max_records=GDELT_MAX_RECORDS_PER_CALL + 1,
        )


@respx.mock
def test_query_window_retries_after_a_429_then_succeeds():
    route = respx.get(GDELT_DOC_API_URL)
    route.side_effect = [
        httpx.Response(429, text="Please limit requests to one every 5 seconds"),
        httpx.Response(200, json=_articles_payload(3)),
    ]

    articles = query_window(
        '"x"', datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 7, 22, tzinfo=timezone.utc)
    )

    assert len(articles) == 3
    assert route.call_count == 2


@respx.mock
def test_query_window_retries_after_http_200_rate_limit_text_then_succeeds():
    """GDELT sometimes signals its rate limit via HTTP 200 + plain-text body
    (confirmed live) instead of a real 429 -- must retry, not hard-fail."""
    route = respx.get(GDELT_DOC_API_URL)
    route.side_effect = [
        httpx.Response(
            200,
            text=(
                "Please limit requests to one every 5 seconds or contact "
                "kalev.leetaru5@gmail.com for larger queries."
            ),
        ),
        httpx.Response(200, json=_articles_payload(3)),
    ]

    articles = query_window(
        '"x"', datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 7, 22, tzinfo=timezone.utc)
    )

    assert len(articles) == 3
    assert route.call_count == 2


@respx.mock
def test_query_window_raises_gdelt_error_after_exhausting_retries():
    respx.get(GDELT_DOC_API_URL).mock(
        return_value=httpx.Response(429, text="rate limited")
    )

    with pytest.raises(GDELTError, match="429"):
        query_window(
            '"x"',
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 22, tzinfo=timezone.utc),
            max_retries=2,
        )


@respx.mock
def test_query_window_raises_gdelt_error_on_non_json_response():
    respx.get(GDELT_DOC_API_URL).mock(
        return_value=httpx.Response(200, text="A maximum of 250 records can be returned.")
    )

    with pytest.raises(GDELTError, match="non-JSON"):
        query_window(
            '"x"', datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 7, 22, tzinfo=timezone.utc)
        )


@respx.mock
def test_query_range_single_window_under_cap_makes_exactly_one_request():
    route = respx.get(GDELT_DOC_API_URL).mock(
        return_value=httpx.Response(200, json=_articles_payload(42))
    )

    articles = query_range(
        '"local school board election"',
        datetime(2026, 7, 19, tzinfo=timezone.utc),
        datetime(2026, 7, 22, tzinfo=timezone.utc),
    )

    assert len(articles) == 42
    assert route.call_count == 1


@respx.mock
def test_query_range_splits_into_sub_windows_past_the_cap_and_combines_over_250():
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 22, tzinfo=timezone.utc)  # 21 days

    def responder(request: httpx.Request) -> httpx.Response:
        params = dict(httpx.QueryParams(request.url.query))
        window_start = datetime.strptime(params["startdatetime"], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        # Top-level (full 21-day) request hits the cap; both halves (each
        # ~10.5 days, > min_window) return under-cap counts that sum >250.
        if window_start == start and params["enddatetime"] == "20260722000000":
            return httpx.Response(200, json=_articles_payload(250))
        if window_start == start:
            return httpx.Response(200, json=_articles_payload(150, domain="first-half.com"))
        return httpx.Response(200, json=_articles_payload(140, domain="second-half.com"))

    route = respx.get(GDELT_DOC_API_URL).mock(side_effect=responder)

    articles = query_range(
        '"climate"',
        start,
        end,
        min_window=timedelta(days=9),
    )

    assert len(articles) > GDELT_MAX_RECORDS_PER_CALL
    assert len(articles) == 150 + 140
    # 1 full-window call (hits cap, triggers the split) + 2 half-window
    # calls (each under the cap, so recursion stops there without needing
    # to reach min_window).
    assert route.call_count == 3


@respx.mock
def test_query_range_stops_recursion_at_min_window_even_if_still_capped():
    respx.get(GDELT_DOC_API_URL).mock(return_value=httpx.Response(200, json=_articles_payload(250)))

    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    articles = query_range('"x"', start, end, min_window=timedelta(hours=2))

    # Range == min_window from the start, so no split ever happens: exactly
    # one request, capped result returned as-is (documented data-loss case).
    assert len(articles) == 250


@respx.mock
def test_fetch_computes_lookback_window_and_builds_query(monkeypatch):
    captured = {}

    def fake_query_range(query, start, end, **kwargs):
        captured["query"] = query
        captured["start"] = start
        captured["end"] = end
        return [{"title": "t", "url": "u", "domain": "d", "published_at": None}]

    monkeypatch.setattr("newsresearch.sourcing.gdelt.query_range", fake_query_range)

    fixed_now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    result = fetch(["climate change"], lookback_days=7, end=fixed_now)

    assert result == [{"title": "t", "url": "u", "domain": "d", "published_at": None}]
    assert captured["query"] == '"climate change"'
    assert captured["end"] == fixed_now
    assert captured["start"] == fixed_now - timedelta(days=7)
