from newsresearch.config import Settings
from newsresearch.sourcing.dedup import (
    dedup,
    dedup_by_title_similarity,
    dedup_by_url,
    normalize_url,
)


class TestNormalizeUrl:
    def test_strips_utm_tracking_params(self):
        assert normalize_url("http://x.com/a?utm_source=y") == normalize_url("http://x.com/a")

    def test_strips_trailing_slash(self):
        assert normalize_url("https://x.com/a/") == normalize_url("https://x.com/a")

    def test_scheme_normalized_http_vs_https(self):
        assert normalize_url("http://x.com/a") == normalize_url("https://x.com/a")

    def test_acceptance_example_utm_and_trailing_slash_and_scheme_all_together(self):
        assert normalize_url("http://x.com/a?utm_source=y") == normalize_url("https://x.com/a/")

    def test_host_is_lowercased(self):
        assert normalize_url("https://X.com/a") == normalize_url("https://x.com/a")

    def test_root_path_trailing_slash_normalizes_same_as_no_path(self):
        assert normalize_url("https://x.com/") == normalize_url("https://x.com")

    def test_non_tracking_query_params_preserved(self):
        assert normalize_url("https://x.com/a?id=5") != normalize_url("https://x.com/a")

    def test_query_param_order_is_normalized(self):
        assert normalize_url("https://x.com/a?b=2&a=1") == normalize_url("https://x.com/a?a=1&b=2")

    def test_multiple_tracking_params_all_stripped(self):
        url = "https://x.com/a?utm_source=y&utm_medium=z&fbclid=abc&id=5"
        assert normalize_url(url) == normalize_url("https://x.com/a?id=5")

    def test_fragment_is_dropped(self):
        assert normalize_url("https://x.com/a#section") == normalize_url("https://x.com/a")

    def test_distinct_paths_remain_distinct(self):
        assert normalize_url("https://x.com/a") != normalize_url("https://x.com/b")


class TestDedupByUrl:
    def test_acceptance_only_one_of_the_two_urls_survives(self):
        articles = [
            {"url": "http://x.com/a?utm_source=y", "title": "Story A", "domain": "x.com"},
            {"url": "https://x.com/a/", "title": "Story A", "domain": "x.com"},
        ]

        deduped = dedup_by_url(articles)

        assert len(deduped) == 1
        assert deduped[0]["url"] == "http://x.com/a?utm_source=y"  # first-seen wins

    def test_distinct_urls_both_survive(self):
        articles = [
            {"url": "https://x.com/a", "title": "Story A", "domain": "x.com"},
            {"url": "https://x.com/b", "title": "Story B", "domain": "x.com"},
        ]

        assert len(dedup_by_url(articles)) == 2

    def test_empty_list_returns_empty_list(self):
        assert dedup_by_url([]) == []


class TestDedupByTitleSimilarity:
    def test_near_identical_titles_different_domains_above_threshold_drops_one(self):
        articles = [
            {"url": "https://reuters.com/a", "title": "Senate Passes Budget Bill", "domain": "reuters.com"},
            {"url": "https://apnews.com/b", "title": "Senate Passes Budget Bill.", "domain": "apnews.com"},
        ]

        deduped = dedup_by_title_similarity(articles, similarity_threshold=0.9)

        assert len(deduped) == 1

    def test_dissimilar_titles_both_survive_below_threshold(self):
        articles = [
            {"url": "https://reuters.com/a", "title": "Senate Passes Budget Bill", "domain": "reuters.com"},
            {"url": "https://apnews.com/b", "title": "City Council Approves New Park", "domain": "apnews.com"},
        ]

        deduped = dedup_by_title_similarity(articles, similarity_threshold=0.9)

        assert len(deduped) == 2

    def test_same_domain_near_identical_titles_are_not_deduped(self):
        # Cross-source dedup only -- same-outlet near-duplicates (e.g. a
        # headline edit) are intentionally left alone by this pass.
        articles = [
            {"url": "https://reuters.com/a", "title": "Senate Passes Budget Bill", "domain": "reuters.com"},
            {"url": "https://reuters.com/b", "title": "Senate Passes Budget Bill.", "domain": "reuters.com"},
        ]

        deduped = dedup_by_title_similarity(articles, similarity_threshold=0.9)

        assert len(deduped) == 2

    def test_explicit_threshold_overrides_settings_default(self):
        articles = [
            {"url": "https://reuters.com/a", "title": "Fire breaks out downtown", "domain": "reuters.com"},
            {"url": "https://apnews.com/b", "title": "Blaze erupts downtown area", "domain": "apnews.com"},
        ]

        # Loosen the threshold enough that these count as duplicates.
        deduped = dedup_by_title_similarity(articles, similarity_threshold=0.5)

        assert len(deduped) == 1

    def test_default_threshold_comes_from_settings_clustering_similarity_threshold(self):
        settings = Settings()
        settings.clustering.similarity_threshold = 0.99  # near-exact match required

        articles = [
            {"url": "https://reuters.com/a", "title": "Senate Passes Budget Bill", "domain": "reuters.com"},
            {"url": "https://apnews.com/b", "title": "Senate Passes the Budget Bill", "domain": "apnews.com"},
        ]

        deduped = dedup_by_title_similarity(articles, settings=settings)

        assert len(deduped) == 2

    def test_empty_list_returns_empty_list(self):
        assert dedup_by_title_similarity([]) == []


class TestDedup:
    def test_combines_url_and_title_passes(self):
        articles = [
            {"url": "http://x.com/a?utm_source=y", "title": "Senate Passes Budget Bill", "domain": "x.com"},
            # exact-URL dupe of the article above
            {"url": "https://x.com/a/", "title": "Senate Passes Budget Bill", "domain": "x.com"},
            # cross-source near-duplicate wire-story title
            {"url": "https://apnews.com/b", "title": "Senate Passes Budget Bill.", "domain": "apnews.com"},
            # unrelated story, should survive
            {"url": "https://bbc.com/c", "title": "City Council Approves New Park", "domain": "bbc.com"},
        ]

        deduped = dedup(articles, similarity_threshold=0.9)

        assert len(deduped) == 2
        assert {a["domain"] for a in deduped} == {"x.com", "bbc.com"}
