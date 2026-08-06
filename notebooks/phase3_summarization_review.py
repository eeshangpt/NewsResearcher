"""Task 3.6.1a -- summarization prompt iteration against real claim clusters.

Real data flow, no synthetic claims:
1. `tests/fixtures/phase3_claim_clustering_corpus.json` (Task 3.3.1a's real
   6-article/120-claim corpus, `claim_text` + metadata only, no article body
   text -- no-full-text-storage rule respected).
2. `clustering/claim_clustering.py::cluster_claims()` run directly (same
   entrypoint `claim_extraction`/`persistence` production code calls) to
   get real cluster assignments (35 clusters, matches Task 3.3.1a's
   documented sweep).
3. `llm/prompts/summarization.txt` run for real via
   `get_chat_model("summarization")` over a sample of clusters chosen to
   cover the three real cluster shapes found in Task 3.3.1a's own
   false-merge-precision finding (0.674, ~1 in 3 wrong merges):
   - a clean multi-source same-fact paraphrase cluster
   - a genuine false-merge cluster with two distinct numeric facts
   - a real single-source cluster HDBSCAN split into multiple *genuinely
     distinct* facts (min_cluster_size=2 tuned for cross-source pairs, not
     single-source multi-fact runs -- a related but different failure mode)

No claim text is ever written to disk beyond what's already in the fixture
JSON (already claim_text-only, no article bodies) -- this script and its
output JSON only ever handle `claim_text` strings, matching the existing
fixture's own no-full-text scope.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from newsresearch.clustering.claim_clustering import cluster_claims
from newsresearch.llm.models import get_chat_model
from newsresearch.llm.schemas import Claim

_HERE = Path(__file__).resolve().parent
_PROMPTS_DIR = _HERE.parent / "newsresearch" / "llm" / "prompts"
_FIXTURE = _HERE.parent / "tests" / "fixtures" / "phase3_claim_clustering_corpus.json"


def load_clusters() -> dict[str, dict]:
    data = json.loads(_FIXTURE.read_text())
    article_claims: dict[str, list[Claim]] = {}
    domain_by_aid: dict[str, str] = {}
    for i, art in enumerate(data):
        aid = f"{art['domain']}::{i}"
        article_claims[aid] = [Claim(**c) for c in art["claims"]]
        domain_by_aid[aid] = art["domain"]

    result = cluster_claims(article_claims)
    clusters = {}
    for label, entry in result["clusters"].items():
        clusters[str(label)] = {
            "claims": [
                {"domain": domain_by_aid[aid], "claim_text": text}
                for aid, text in zip(entry["article_ids"], entry["claim_text"])
            ],
        }
    return clusters


def format_claims(claims: list[dict]) -> str:
    return "\n".join(f"- {c['domain']}: {c['claim_text']}" for c in claims)


def summarize(claims: list[dict]) -> str:
    template_text = (_PROMPTS_DIR / "summarization.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("summarization")
    chain = prompt | model
    result = chain.invoke({"claims": format_claims(claims)})
    return result.content


def main() -> None:
    clusters = load_clusters()

    # Chosen for real, distinct failure/success modes -- see module docstring.
    sample_labels = [
        "21",  # clean 3-source same-fact paraphrase
        "31",  # 4 "articles" (3 asserting) mixing a shared fact + one extra
        "2",  # genuine false-merge: two different price cuts, same article
        "9",  # single-source, 5 genuinely distinct benchmark-score facts
        "34",  # single-source, 2 related-but-distinct claims
        "0",  # single-source, 2 claims about the same PLA unit/model
        "15",  # 3 sources, each contributing one genuinely distinct fact
        "1",  # single-source, includes a named-third-party-attributed opinion claim
    ]

    out = {}
    for label in sample_labels:
        entry = clusters[label]
        summary = summarize(entry["claims"])
        out[label] = {"claims": entry["claims"], "summary": summary}
        print(f"\n=== cluster {label} ({len(entry['claims'])} claims) ===")
        for c in entry["claims"]:
            print(f"  [{c['domain']}] {c['claim_text']}")
        print(f"  -> SUMMARY: {summary}")

    (_HERE / "phase3_summarization_samples.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
