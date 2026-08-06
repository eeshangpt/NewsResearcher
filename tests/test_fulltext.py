"""Task 3.1.1 tests: in-memory full-text fetch, no persistence path."""

import ast
import socket
from pathlib import Path
from unittest.mock import patch

from newsresearch.config import Settings
from newsresearch.sourcing.fulltext import (
    _trafilatura_config,
    fetch_fulltext,
    fetch_fulltext_for_cluster,
)

FULLTEXT_PATH = Path(__file__).resolve().parent.parent / "newsresearch" / "sourcing" / "fulltext.py"


def test_fulltext_module_never_imports_persistence():
    tree = ast.parse(FULLTEXT_PATH.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    assert not any("persistence" in name for name in names)


def test_fetch_fulltext_success():
    with (
        patch("trafilatura.fetch_url", return_value="<html>raw</html>"),
        patch("trafilatura.extract", return_value="the article body"),
    ):
        assert fetch_fulltext("https://example.com/a") == "the article body"


def test_fetch_fulltext_unreachable_url_soft_fails():
    with patch("trafilatura.fetch_url", side_effect=Exception("network error")):
        assert fetch_fulltext("https://unreachable.example.com/a") is None


def test_fetch_fulltext_no_content_downloaded():
    with patch("trafilatura.fetch_url", return_value=None):
        assert fetch_fulltext("https://example.com/a") is None


def test_fetch_fulltext_empty_extraction_result():
    with (
        patch("trafilatura.fetch_url", return_value="<html>raw</html>"),
        patch("trafilatura.extract", return_value=None),
    ):
        assert fetch_fulltext("https://example.com/a") is None


def test_fetch_fulltext_timeout_soft_fails_and_never_hangs():
    """Issue #108: a `trafilatura.fetch_url()` timeout must return `None`,
    not raise or block indefinitely (it internally raises `socket.timeout`
    on a hung download, which `fetch_fulltext`'s `except Exception` catches)."""
    with patch("trafilatura.fetch_url", side_effect=socket.timeout("timed out")):
        assert fetch_fulltext("https://slow.example.com/a") is None


def test_fetch_fulltext_passes_configured_timeout_to_trafilatura():
    settings = Settings()
    settings.sourcing.fulltext_fetch_timeout_seconds = 3.0

    with patch("trafilatura.fetch_url", return_value=None) as mock_fetch_url:
        fetch_fulltext("https://example.com/a", settings)

    _, kwargs = mock_fetch_url.call_args
    assert kwargs["config"].get("DEFAULT", "DOWNLOAD_TIMEOUT") == "3"


def test_trafilatura_config_download_timeout_is_getint_parseable():
    """Regression for tech-lead review: `trafilatura` reads this key via
    `config.getint(...)`; writing a float-formatted string ("30.0") makes
    `getint` raise, which trafilatura swallows -- silently returning `None`
    for every fetch regardless of reachability. Build the real `ConfigParser`
    (no mocking `fetch_url`) and exercise the exact read path trafilatura uses."""
    config = _trafilatura_config(30.0)
    assert config.getint("DEFAULT", "DOWNLOAD_TIMEOUT") == 30


def test_fetch_fulltext_for_cluster_one_bad_url_does_not_crash_batch():
    articles = [
        {"url": "https://good.example.com/a", "title": "t1", "domain": "good.example.com"},
        {"url": "https://bad.example.com/b", "title": "t2", "domain": "bad.example.com"},
    ]

    def fake_fetch_url(url, *args, **kwargs):
        if "bad" in url:
            raise Exception("network error")
        return "<html>raw</html>"

    with (
        patch("trafilatura.fetch_url", side_effect=fake_fetch_url),
        patch("trafilatura.extract", return_value="body text"),
    ):
        results = fetch_fulltext_for_cluster(articles)

    assert results == [
        {"url": "https://good.example.com/a", "fulltext": "body text"},
        {"url": "https://bad.example.com/b", "fulltext": None},
    ]