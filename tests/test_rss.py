import textwrap

import httpx
import respx
import yaml
from freezegun import freeze_time

from newsresearch.sourcing import rss


def _feed_xml(items: str) -> bytes:
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
        <title>Test Feed</title>
        {items}
        </channel></rss>
        """
    ).encode()


BBC_FEED = _feed_xml(
    """
    <item>
      <title>Ukraine ceasefire talks resume in Geneva</title>
      <description>Diplomats meet again this week.</description>
      <link>https://www.bbc.co.uk/news/articles/aaa</link>
      <pubDate>Wed, 15 Jul 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Local football result roundup</title>
      <description>Weekend scores from the league.</description>
      <link>https://www.bbc.co.uk/news/articles/bbb</link>
      <pubDate>Wed, 15 Jul 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Ukraine aid package approved</title>
      <description>Old story, outside the lookback window.</description>
      <link>https://www.bbc.co.uk/news/articles/ccc</link>
      <pubDate>Thu, 01 Jan 2026 10:00:00 GMT</pubDate>
    </item>
    """
)

GUARDIAN_FEED = _feed_xml(
    """
    <item>
      <title>Ukraine reconstruction fund debated at summit</title>
      <description>European leaders discuss financing.</description>
      <link>https://www.theguardian.com/world/2026/jul/16/ukraine-fund</link>
      <pubDate>Thu, 16 Jul 2026 08:00:00 GMT</pubDate>
    </item>
    """
)

FEEDS_UNDER_TEST = {
    "bbc.com": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "theguardian.com": "https://www.theguardian.com/world/rss",
}


def test_load_trusted_domains_returns_domain_to_tier_dict():
    domains = rss.load_trusted_domains()

    assert isinstance(domains, dict)
    assert domains["reuters.com"] == "wire"


def test_outlet_rss_feeds_only_maps_known_trusted_domains():
    trusted_domains = yaml.safe_load(rss.TRUSTED_OUTLETS_PATH.read_text())

    assert len(rss.OUTLET_RSS_FEEDS) >= 2
    for domain in rss.OUTLET_RSS_FEEDS:
        assert domain in trusted_domains


@respx.mock
@freeze_time("2026-07-22 12:00:00")
def test_fetch_trusted_rss_filters_by_keyword_and_lookback_across_two_feeds():
    respx.get(FEEDS_UNDER_TEST["bbc.com"]).mock(return_value=httpx.Response(200, content=BBC_FEED))
    respx.get(FEEDS_UNDER_TEST["theguardian.com"]).mock(
        return_value=httpx.Response(200, content=GUARDIAN_FEED)
    )

    articles = rss.fetch_trusted_rss("ukraine", lookback_days=10, feeds=FEEDS_UNDER_TEST)

    urls = {article["url"] for article in articles}
    domains = {article["domain"] for article in articles}
    assert urls == {
        "https://www.bbc.co.uk/news/articles/aaa",
        "https://www.theguardian.com/world/2026/jul/16/ukraine-fund",
    }
    assert domains == {"bbc.com", "theguardian.com"}
    assert all(article["source_type"] == "rss" for article in articles)
    assert all(article["published_at"] is not None for article in articles)


@respx.mock
@freeze_time("2026-07-22 12:00:00")
def test_fetch_trusted_rss_drops_non_matching_keyword_and_stale_entries():
    respx.get(FEEDS_UNDER_TEST["bbc.com"]).mock(return_value=httpx.Response(200, content=BBC_FEED))
    respx.get(FEEDS_UNDER_TEST["theguardian.com"]).mock(
        return_value=httpx.Response(200, content=GUARDIAN_FEED)
    )

    articles = rss.fetch_trusted_rss("ukraine", lookback_days=10, feeds=FEEDS_UNDER_TEST)
    urls = {article["url"] for article in articles}

    assert "https://www.bbc.co.uk/news/articles/bbb" not in urls  # keyword mismatch
    assert "https://www.bbc.co.uk/news/articles/ccc" not in urls  # outside lookback window


@respx.mock
@freeze_time("2026-07-22 12:00:00")
def test_fetch_trusted_rss_accepts_a_list_of_keywords_matched_with_or_semantics():
    respx.get(FEEDS_UNDER_TEST["bbc.com"]).mock(return_value=httpx.Response(200, content=BBC_FEED))
    respx.get(FEEDS_UNDER_TEST["theguardian.com"]).mock(
        return_value=httpx.Response(200, content=GUARDIAN_FEED)
    )

    articles = rss.fetch_trusted_rss(
        ["football", "nonexistent-keyword"], lookback_days=10, feeds=FEEDS_UNDER_TEST
    )

    urls = {article["url"] for article in articles}
    assert urls == {"https://www.bbc.co.uk/news/articles/bbb"}


@respx.mock
@freeze_time("2026-07-22 12:00:00")
def test_fetch_trusted_rss_skips_a_failing_feed_and_still_returns_the_others():
    respx.get(FEEDS_UNDER_TEST["bbc.com"]).mock(return_value=httpx.Response(500))
    respx.get(FEEDS_UNDER_TEST["theguardian.com"]).mock(
        return_value=httpx.Response(200, content=GUARDIAN_FEED)
    )

    articles = rss.fetch_trusted_rss("ukraine", lookback_days=10, feeds=FEEDS_UNDER_TEST)

    assert len(articles) == 1
    assert articles[0]["domain"] == "theguardian.com"
