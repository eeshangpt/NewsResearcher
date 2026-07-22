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


@respx.mock
def test_fetch_google_news_backfill_parses_title_domain_and_published_date():
    respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(return_value=httpx.Response(200, content=GOOGLE_NEWS_FEED))

    articles = gnb.fetch_google_news_backfill("ukraine", lookback_days=7)

    assert len(articles) == 2
    first = articles[0]
    assert first["title"] == "Ceasefire talks resume in Geneva"
    assert first["domain"] == "reuters.com"
    assert first["publisher_name"] == "Reuters"
    assert first["source_type"] == "google_news_backfill"
    assert first["url"] == "https://news.google.com/rss/articles/redirect-token-1?oc=5"
    assert first["published_at"] is not None


@respx.mock
def test_fetch_google_news_backfill_accepts_a_list_of_keywords():
    request = respx.get(gnb.GOOGLE_NEWS_RSS_URL).mock(
        return_value=httpx.Response(200, content=GOOGLE_NEWS_FEED)
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


def test_domain_from_url_strips_www_prefix():
    assert gnb._domain_from_url("https://www.reuters.com/world") == "reuters.com"
    assert gnb._domain_from_url("https://apnews.com/article") == "apnews.com"
    assert gnb._domain_from_url(None) is None
