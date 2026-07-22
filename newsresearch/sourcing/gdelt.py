"""GDELT DOC 2.0 API client (TRD 4.2 Sourcing Agent, Story 1.2).

Free, no-API-key keyword+date-range search against the real GDELT DOC 2.0
API (https://api.gdeltproject.org/api/v2/doc/doc) in JSON article-list mode.

Two things the live API forces this module to handle explicitly:

- **A hard 250-record cap per call.** GDELT doesn't offer a cursor/offset
  parameter for this endpoint -- the only way to get more than 250 results
  for a broad keyword/date-range combination is to split the range into
  narrower sub-windows and query each separately. `query_range()` detects
  this by noticing a sub-window's result count hit exactly the cap (a
  strong signal the true count is truncated, since GDELT gives no total-hits
  field) and recursively bisects that window until either the count drops
  below the cap or `min_window` is reached.
- **Aggressive rate-limiting, signaled two different ways.** GDELT
  documents a "one request per 5 seconds" policy and returns a real HTTP 429
  when violated, but -- also confirmed against the live API while building
  this module -- sometimes instead returns HTTP **200** with a plain-text
  ("Please limit requests to one every 5 seconds...") body rather than a
  429 or JSON. `query_window()` treats both as the same retryable
  rate-limit condition and backs off exponentially rather than treating the
  200-with-text-body case as a hard parse failure.

Every returned article carries `"source_type": "gdelt"`, matching the
convention `rss.py`/`google_news_backfill.py` already use (Story 1.3) so
`reputation/signals.py`'s presence-frequency signal (Task 1.5.1) can key off
`(domain, source_type)` across all three sourcing modules uniformly.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Confirmed against the live API: requesting more than this raises a
# plain-text "A maximum of 250 records can be returned." error even though
# GDELT responds with HTTP 200 for it, not a JSON body.
GDELT_MAX_RECORDS_PER_CALL = 250

_GDELT_DATETIME_FORMAT = "%Y%m%d%H%M%S"
_GDELT_SEENDATE_FORMAT = "%Y%m%dT%H%M%SZ"

# Substring GDELT's plain-text rate-limit response contains, e.g. "Please
# limit requests to one every 5 seconds or contact ...". Checked
# case-insensitively against a 200 response's body -- GDELT's rate limit
# isn't always signaled via a real HTTP 429 (confirmed live).
_RATE_LIMIT_TEXT_MARKER = "limit requests"

# GDELT's own documented rate limit is "one request per 5 seconds"; observed
# in practice to sometimes require longer. Used both as the default delay
# between sequential sub-window requests and as the base of the 429
# exponential-backoff schedule.
DEFAULT_INTER_REQUEST_DELAY_SECONDS = 5.0


class GDELTError(Exception):
    """Raised when the GDELT DOC 2.0 API returns a non-retryable error."""


def _format_gdelt_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_GDELT_DATETIME_FORMAT)


def _build_query(keywords: list[str]) -> str:
    """Join a keyword list into a GDELT DOC 2.0 boolean-OR query string.

    Verified directly against the real GDELT DOC 2.0 endpoint (tech-lead
    review of Task 1.10.1): quoting a single-token keyword (e.g. `"Iran"`)
    gets a real HTTP 200 with the plain-text body "The specified phrase is
    too short" -- GDELT's phrase-quoting is for multi-word phrases only, a
    lone token must be bare. Only whitespace-containing keywords are quoted
    here. Separately, an OR'd multi-term query gets a real HTTP 200 with
    "Queries containing OR'd terms must be surrounded by ()" unless the
    whole OR expression is parenthesized -- so terms are only OR-joined
    (with wrapping parens) when there's more than one keyword.
    """
    if not keywords:
        raise ValueError("keywords must be a non-empty list")
    terms = [f'"{keyword}"' if " " in keyword else keyword for keyword in keywords]
    if len(terms) == 1:
        return terms[0]
    return "(" + " OR ".join(terms) + ")"


def _parse_article(raw: dict) -> dict:
    """Map a raw GDELT `articles[]` entry to
    `{title, url, domain, published_at, source_type}`."""
    published_at = None
    seendate = raw.get("seendate")
    if seendate:
        try:
            published_at = datetime.strptime(seendate, _GDELT_SEENDATE_FORMAT).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            logger.warning("gdelt: unparseable seendate %r; leaving published_at=None", seendate)

    domain = raw.get("domain")
    if domain:
        # Defensive normalization -- GDELT's own `domain` values are already
        # clean (lowercase, no `www.`), but don't assume that holds forever.
        domain = domain.strip().lower()

    return {
        "title": raw.get("title"),
        "url": raw.get("url"),
        "domain": domain,
        "published_at": published_at,
        "source_type": "gdelt",
    }


def _is_rate_limited(response: httpx.Response) -> bool:
    """True for a real HTTP 429, or GDELT's observed HTTP-200-with-plain-text
    rate-limit body -- both are the same retryable condition."""
    if response.status_code == 429:
        return True
    return _RATE_LIMIT_TEXT_MARKER in response.text.lower()


def _retry_after_seconds(response: httpx.Response, attempt: int, base_delay: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return base_delay * (2 ** (attempt - 1))


def query_window(
    query: str,
    start: datetime,
    end: datetime,
    *,
    max_records: int = GDELT_MAX_RECORDS_PER_CALL,
    client: httpx.Client | None = None,
    max_retries: int = 5,
    backoff_seconds: float = DEFAULT_INTER_REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Query one GDELT DOC 2.0 window (<= 250 records) in JSON article-list mode.

    `query` is a raw GDELT query string (e.g. built via `_build_query()`).
    `start`/`end` bound the `startdatetime`/`enddatetime` params (UTC).
    Retries with exponential backoff (`backoff_seconds` doubling per
    attempt, capped at `max_retries`) on either a real HTTP 429 or GDELT's
    HTTP-200-with-plain-text-rate-limit-body quirk (see module docstring),
    honoring a `Retry-After` header if the response includes one.

    Raises `ValueError` if `max_records` exceeds GDELT's 250-record cap
    (use `query_range()` to page over sub-windows instead), and `GDELTError`
    if retries are exhausted or the API returns a non-JSON, non-rate-limit
    body.
    """
    if max_records > GDELT_MAX_RECORDS_PER_CALL:
        raise ValueError(
            f"GDELT DOC 2.0 caps maxrecords at {GDELT_MAX_RECORDS_PER_CALL} per call; "
            "use query_range() to page over sub-windows for larger expected volumes."
        )

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": _format_gdelt_datetime(start),
        "enddatetime": _format_gdelt_datetime(end),
        "sort": "datedesc",
    }

    owns_client = client is None
    http_client = client or httpx.Client(timeout=30.0)
    try:
        response: httpx.Response | None = None
        for attempt in range(1, max_retries + 2):
            response = http_client.get(GDELT_DOC_API_URL, params=params)
            if not _is_rate_limited(response):
                break
            if attempt > max_retries:
                raise GDELTError(
                    f"GDELT DOC 2.0 API still rate-limited after {max_retries} retries "
                    f"for window {start.isoformat()}..{end.isoformat()} "
                    f"(last status {response.status_code})"
                )
            delay = _retry_after_seconds(response, attempt, backoff_seconds)
            logger.warning(
                "gdelt: rate-limited (status=%d, attempt %d/%d), backing off %.1fs",
                response.status_code,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)

        assert response is not None  # loop always runs at least once
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise GDELTError(
                f"GDELT DOC 2.0 API returned a non-JSON response: {response.text[:200]!r}"
            ) from exc

        return [_parse_article(raw) for raw in payload.get("articles", [])]
    finally:
        if owns_client:
            http_client.close()


def query_range(
    query: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.Client | None = None,
    max_retries: int = 5,
    backoff_seconds: float = DEFAULT_INTER_REQUEST_DELAY_SECONDS,
    inter_request_delay: float = DEFAULT_INTER_REQUEST_DELAY_SECONDS,
    min_window: timedelta = timedelta(days=1),
) -> list[dict]:
    """Query a date range, paginating over sub-windows past GDELT's 250-record cap.

    Queries the full `start`..`end` range in one call first. If (and only
    if) that call returns exactly `GDELT_MAX_RECORDS_PER_CALL` results --
    the only signal GDELT gives that a window's true count was truncated,
    since the API returns no total-hits field -- the window is bisected and
    each half is queried (and recursively bisected under the same rule)
    until sub-windows return under the cap or shrink to `min_window`.

    `inter_request_delay` is slept between sequential sub-window requests to
    respect GDELT's documented rate limit; `query_window()`'s own 429
    backoff is the fallback if that's not sufficient in practice.

    Note: adjacent sub-windows share their bisection instant as one
    window's `enddatetime` and the next's `startdatetime`, so an article
    seen at exactly that instant could in principle be returned by both
    halves. Not de-duplicated here -- `dedup.py`'s exact-URL pass (Story
    1.4), which every sourcing module's output already needs regardless,
    absorbs this.
    """
    owns_client = client is None
    http_client = client or httpx.Client(timeout=30.0)
    try:
        return _query_range_recursive(
            query,
            start,
            end,
            client=http_client,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            inter_request_delay=inter_request_delay,
            min_window=min_window,
        )
    finally:
        if owns_client:
            http_client.close()


def _query_range_recursive(
    query: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.Client,
    max_retries: int,
    backoff_seconds: float,
    inter_request_delay: float,
    min_window: timedelta,
) -> list[dict]:
    articles = query_window(
        query, start, end, client=client, max_retries=max_retries, backoff_seconds=backoff_seconds
    )

    hit_cap = len(articles) >= GDELT_MAX_RECORDS_PER_CALL
    if not hit_cap or (end - start) <= min_window:
        if hit_cap:
            logger.warning(
                "gdelt: window %s..%s still hit the %d-record cap at the minimum "
                "splittable window size (min_window=%s); some articles in this "
                "window were not retrieved.",
                start.isoformat(),
                end.isoformat(),
                GDELT_MAX_RECORDS_PER_CALL,
                min_window,
            )
        return articles

    midpoint = start + (end - start) / 2
    logger.info(
        "gdelt: window %s..%s hit the %d-record cap, splitting at %s",
        start.isoformat(),
        end.isoformat(),
        GDELT_MAX_RECORDS_PER_CALL,
        midpoint.isoformat(),
    )

    time.sleep(inter_request_delay)
    first_half = _query_range_recursive(
        query,
        start,
        midpoint,
        client=client,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        inter_request_delay=inter_request_delay,
        min_window=min_window,
    )
    time.sleep(inter_request_delay)
    second_half = _query_range_recursive(
        query,
        midpoint,
        end,
        client=client,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        inter_request_delay=inter_request_delay,
        min_window=min_window,
    )
    return first_half + second_half


def fetch(
    keywords: list[str],
    lookback_days: int,
    *,
    end: datetime | None = None,
    client: httpx.Client | None = None,
    max_retries: int = 5,
    backoff_seconds: float = DEFAULT_INTER_REQUEST_DELAY_SECONDS,
    inter_request_delay: float = DEFAULT_INTER_REQUEST_DELAY_SECONDS,
    min_window: timedelta = timedelta(days=1),
) -> list[dict]:
    """Fetch GDELT articles for `keywords` over the last `lookback_days` days.

    Convenience entrypoint for callers (e.g. the future Sourcing Agent,
    Task 1.9) that just have a keyword list and a lookback window, not a raw
    GDELT query string or explicit date range. Builds the OR-joined query
    and delegates paging/backoff to `query_range()`.
    """
    window_end = end or datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=lookback_days)
    query = _build_query(keywords)
    return query_range(
        query,
        window_start,
        window_end,
        client=client,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        inter_request_delay=inter_request_delay,
        min_window=min_window,
    )
