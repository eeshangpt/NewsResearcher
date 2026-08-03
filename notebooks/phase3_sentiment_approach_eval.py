"""Task 3.4.1a -- lexicon vs. small-model sentiment, tested on real claims/articles.

Runs VADER (`vaderSentiment`, not yet a project dependency -- installed
on-the-fly via `uv run --with vaderSentiment`, nothing added to pyproject.toml
by this analysis-only script) over:
  1. real extracted claim_text strings from notebooks/phase3_claim_extraction_samples.json
     (Task 3.2.1a's output -- already a committed fixture, no new full-text storage)
  2. a couple of real articles' full text, fetched transiently in-memory via
     `sourcing/fulltext.py::fetch_fulltext` (never written to disk -- this
     script only ever prints a short scored excerpt, per the no-full-text-
     storage rule)

Usage:
    uv run --with vaderSentiment python notebooks/phase3_sentiment_approach_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLES_PATH = Path(__file__).resolve().parent / "phase3_claim_extraction_samples.json"

# Standard VADER-recommended compound thresholds for a 3-class bucketing.
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def label_for(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "positive"
    if compound <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score_claims(analyzer: SentimentIntensityAnalyzer) -> list[dict]:
    articles = json.loads(SAMPLES_PATH.read_text())
    out = []
    for article in articles:
        for claim in article["claims"]:
            compound = analyzer.polarity_scores(claim["claim_text"])["compound"]
            out.append(
                {
                    "domain": article["domain"],
                    "claim_text": claim["claim_text"],
                    "compound": round(compound, 4),
                    "label": label_for(compound),
                }
            )
    return out


def score_articles_live(analyzer: SentimentIntensityAnalyzer, urls: list[str]) -> None:
    """Fetch real article fulltext transiently, score it, print a short
    excerpt + score. Fulltext is never written to disk -- only kept in a
    local variable for the duration of this function call."""
    from newsresearch.sourcing.fulltext import fetch_fulltext

    for url in urls:
        text = fetch_fulltext(url)
        if not text:
            print(f"[skip] no fulltext for {url}")
            continue
        compound = analyzer.polarity_scores(text)["compound"]
        excerpt = text[:160].replace("\n", " ")
        print(f"{label_for(compound):8s} compound={compound:+.4f}  {url}")
        print(f"          excerpt: {excerpt}...")
        # `text` goes out of scope here -- never persisted.


def demo() -> None:
    """Self-check: known-polarity strings must land in the expected bucket."""
    analyzer = SentimentIntensityAnalyzer()
    cases = {
        "Dozens of people have been killed and shot in the clashes.": "negative",
        "New innovations could dramatically improve lives and save people.": "positive",
        "The meeting is scheduled for Monday afternoon.": "neutral",
    }
    for text, expected in cases.items():
        compound = analyzer.polarity_scores(text)["compound"]
        got = label_for(compound)
        assert got == expected, f"{text!r} -> {got}, expected {expected}"
    print("demo() self-check passed.")


if __name__ == "__main__":
    analyzer = SentimentIntensityAnalyzer()
    demo()

    print("\n=== claim-level sentiment (real claims from Task 3.2.1a samples) ===")
    scored = score_claims(analyzer)
    counts: dict[str, int] = {}
    for row in scored:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"n={len(scored)} label distribution: {counts}")

    # Print a representative sample per label.
    for label in ("positive", "negative", "neutral"):
        print(f"\n-- sample '{label}' claims --")
        for row in [r for r in scored if r["label"] == label][:4]:
            print(f"  [{row['compound']:+.4f}] ({row['domain']}) {row['claim_text'][:140]}")

    print("\n=== article-level sentiment (real fulltext, fetched transiently) ===")
    sample_urls = [
        "https://www.bbc.co.uk/news/articles/c5yvqk69enko?at_medium=RSS&at_campaign=rss",  # Kashmir protests/deaths
        "https://www.theguardian.com/society/2026/jul/31/hiv-aids-medication-usaid-cuts",  # HIV funding
    ]
    score_articles_live(analyzer, sample_urls)