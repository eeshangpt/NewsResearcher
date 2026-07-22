"""Real-API smoke tests for `sourcing/gdelt.py` (Story 1.2).

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach and `tests/live/test_langfuse_live.py`
for the existing precedent). Hits the real GDELT DOC 2.0 API, which is free
and requires no API key, but is aggressively rate-limited (observed while
building this module to sometimes need well over its documented "one
request per 5 seconds" policy) -- these tests use generous backoff/retry
settings and can take a couple of minutes to run for that reason.

Run explicitly:

    uv run pytest -m live tests/live/test_gdelt_live.py
"""

from datetime import timedelta

import pytest

from newsresearch.sourcing.gdelt import GDELT_MAX_RECORDS_PER_CALL, fetch

pytestmark = pytest.mark.live


def test_single_window_real_query_returns_parsed_articles_under_cap():
    """Task 1.2.1: a real GDELT call for a narrow, low-volume keyword+range
    returns parsed article dicts, under the 250-record cap."""
    articles = fetch(
        ["artificial intelligence regulation"],
        lookback_days=2,
        max_retries=8,
        backoff_seconds=8.0,
        min_window=timedelta(days=1),
    )

    assert len(articles) < GDELT_MAX_RECORDS_PER_CALL
    for article in articles:
        assert article["url"]
        assert article["domain"]
        assert article["title"]
        # published_at may be None for a handful of malformed seendate
        # values, but should be populated for the overwhelming majority.
    if articles:
        with_dates = [a for a in articles if a["published_at"] is not None]
        assert len(with_dates) / len(articles) > 0.9


def test_broad_multiweek_query_exceeds_cap_via_real_sub_window_pagination():
    """Task 1.2.2: a genuinely broad keyword ("climate") over a 3-week
    lookback is known to exceed GDELT's 250-record cap for a single window
    (confirmed live while building this module -- a plain 21-day query for
    "climate" alone returns exactly 250, the cap-hit signal). `fetch()`
    should detect that and combine sub-window results to exceed 250.

    `min_window=9 days` bounds this to a small, deterministic number of real
    API calls (1 full-range check + 2 half-range calls) rather than
    recursing all the way to day/hour granularity, which would multiply
    the real request count (and this test's runtime) substantially given
    GDELT's rate limiting.
    """
    articles = fetch(
        ["climate"],
        lookback_days=21,
        max_retries=8,
        backoff_seconds=8.0,
        inter_request_delay=8.0,
        min_window=timedelta(days=9),
    )

    assert len(articles) > GDELT_MAX_RECORDS_PER_CALL
