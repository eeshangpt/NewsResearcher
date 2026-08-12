import json
import textwrap

import httpx
import pytest
import respx

from newsresearch.sourcing import google_news_backfill as gnb

GOOGLE_NEWS_FEED = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
    <title>"ukraine" - Google News</title>
    <item>
      <title>Ceasefire talks resume in Geneva - Reuters</title>
      <link>https://news.google.com/rss/articles/redirect-token-1?oc=5</link>
      <pubDate>Wed, 15 Jul 2026 10:00:00 GMT</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
    </item>
    <item>
      <title>Aid package approved by parliament - Example News</title>
      <link>https://news.google.com/rss/articles/redirect-token-2?oc=5</link>
      <pubDate>Thu, 16 Jul 2026 08:00:00 GMT</pubDate>
      <source url="https://www.example-news.com">Example News</source>
    </item>
    </channel></rss>
    """
).encode()


def _shell_page_html(signature: str, timestamp: str) -> str:
    return f'<c-wiz><div jscontroller="x" data-n-a-sg="{signature}" data-n-a-ts="{timestamp}"></div></c-wiz>'


def _batchexecute_body(real_url: str) -> str:
    # Matches the real batchexecute response shape: a `)]}'` anti-hijacking
    # prefix, then one JSON line with the RPC result plus trailing footer
    # rows that `_resolve_real_article_url`'s `[:-2]` slices off.
    rpc_result = ["wrb.fr", "Fbv4je", json.dumps(["garturlres", real_url, 1]), None, None, None, ""]
    body = json.dumps([rpc_result, ["di", 12], ["af.httprm", 12, "token", 8]])
    return f")]}}'\n\n{body}"


def _mock_resolutions(redirect_to_real_url: dict[str, str], *, signature="sig", timestamp="123"):
    """Mock the shell-page GET for each redirect URL, plus a single shared
    batchexecute POST route that dispatches on the base64 article id
    embedded in the request body (respx matches routes by URL only, so a
    per-article response needs one shared route with a `side_effect`)."""
    for redirect_url in redirect_to_real_url:
        respx.get(redirect_url).mock(
            return_value=httpx.Response(200, text=_shell_page_html(signature, timestamp))
        )

    redirect_by_base64_id = {
        gnb._base64_id_from_redirect_url(url): real_url
        for url, real_url in redirect_to_real_url.items()
    }

    def _dispatch(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        for base64_id, real_url in redirect_by_base64_id.items():
            if base64_id in body:
                return httpx.Response(200, text=_batchexecute_body(real_url))
        return httpx.Response(200, text=_batchexecute_body("https://unmatched.example.com"))

    respx.post(gnb._GOOGLE_NEWS_BATCHEXECUTE_URL).mock(side_effect=_dispatch)


@respx.mock
def test_fetch_google_news_backfill_parses_title_domain_and_resolves_real_url():
    respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(return_value=httpx.Response(200, content=GOOGLE_NEWS_FEED))
    _mock_resolutions(
        {
            "https://news.google.com/rss/articles/redirect-token-1?oc=5": (
                "https://www.reuters.com/world/ceasefire-talks-resume"
            ),
            "https://news.google.com/rss/articles/redirect-token-2?oc=5": (
                "https://www.example-news.com/aid-package"
            ),
        }
    )

    articles = gnb.fetch_google_news_backfill("ukraine", lookback_days=7)

    assert len(articles) == 2
    first = articles[0]
    assert first["title"] == "Ceasefire talks resume in Geneva"
    assert first["domain"] == "reuters.com"
    assert first["publisher_name"] == "Reuters"
    assert first["source_type"] == "google_news_backfill"
    assert first["url"] == "https://www.reuters.com/world/ceasefire-talks-resume"
    assert first["published_at"] is not None


@respx.mock
def test_fetch_google_news_backfill_drops_entry_when_real_url_unresolvable():
    respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(return_value=httpx.Response(200, content=GOOGLE_NEWS_FEED))
    # First entry's shell page has no data-n-a-sg/ts attributes -- unresolvable.
    respx.get("https://news.google.com/rss/articles/redirect-token-1?oc=5").mock(
        return_value=httpx.Response(200, text="<html>no signature here</html>")
    )
    _mock_resolutions(
        {
            "https://news.google.com/rss/articles/redirect-token-2?oc=5": (
                "https://www.example-news.com/aid-package"
            )
        }
    )

    articles = gnb.fetch_google_news_backfill("ukraine", lookback_days=7)

    assert len(articles) == 1
    assert articles[0]["url"] == "https://www.example-news.com/aid-package"


@respx.mock
def test_fetch_google_news_backfill_accepts_a_list_of_keywords():
    request = respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(
        return_value=httpx.Response(200, content=GOOGLE_NEWS_FEED)
    )
    _mock_resolutions(
        {
            "https://news.google.com/rss/articles/redirect-token-1?oc=5": (
                "https://www.reuters.com/world/ceasefire-talks-resume"
            ),
            "https://news.google.com/rss/articles/redirect-token-2?oc=5": (
                "https://www.example-news.com/aid-package"
            ),
        }
    )

    gnb.fetch_google_news_backfill(["ukraine", "ceasefire"], lookback_days=7)

    sent_request = request.calls.last.request
    query = httpx.QueryParams(sent_request.url.query)
    assert query["q"] == "ukraine ceasefire when:7d"


@respx.mock
def test_fetch_google_news_backfill_raises_on_http_error_no_internal_soft_fail():
    respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        gnb.fetch_google_news_backfill("ukraine")


@respx.mock
def test_resolve_real_article_url_soft_fails_on_batchexecute_error():
    respx.get("https://news.google.com/rss/articles/token?oc=5").mock(
        return_value=httpx.Response(200, text=_shell_page_html("sig", "123"))
    )
    respx.post(gnb._GOOGLE_NEWS_BATCHEXECUTE_URL).mock(return_value=httpx.Response(500))

    result = gnb._resolve_real_article_url(
        "https://news.google.com/rss/articles/token?oc=5", timeout=5.0
    )

    assert result is None


def test_base64_id_from_redirect_url():
    assert (
        gnb._base64_id_from_redirect_url("https://news.google.com/rss/articles/ABC123?oc=5")
        == "ABC123"
    )
    assert gnb._base64_id_from_redirect_url("https://example.com/not-google-news") is None


def test_domain_from_url_strips_www_prefix():
    assert gnb._domain_from_url("https://www.reuters.com/world") == "reuters.com"
    assert gnb._domain_from_url("https://apnews.com/article") == "apnews.com"
    assert gnb._domain_from_url(None) is None


def test_domain_from_url_lowercases_mixed_case_host():
    # Story 1.9 integration checkpoint: gdelt.py/rss.py always emit a
    # lowercase, no-`www.` domain -- this module must match that normalized
    # form so dedup.py's same-domain string-equality guard doesn't misfire.
    assert gnb._domain_from_url("https://WWW.Reuters.COM/world") == "reuters.com"
