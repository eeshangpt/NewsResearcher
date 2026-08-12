"""Task 4.1.1a -- run `llm/prompts/bias_framing.txt` against real persisted clusters.

Reads the real `task371-leipzig-drone` subtopic out of local Postgres
(`NEWSRESEARCH_DATABASE_URL`) via the production
`persistence/claim_clusters.py::read_cluster_article_relations`, batches
clusters `bias_framing_batch_size` at a time (tech-lead's 2026-08-12
decision, default 8) and calls the real `get_chat_model("bias_framing")`.

Only `claim_text` + `article_id` + `domain` ever leave the DB -- no article
full text is touched, fetched, or written anywhere (FR-25).

Usage:
    uv run python notebooks/phase4_bias_framing_review.py [--repeat N] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so this runs from the repo root

from langchain_core.prompts import ChatPromptTemplate

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.persistence.claim_clusters import read_cluster_article_relations
from newsresearch.persistence.db import init_db

from phase4_bias_framing_schema_draft import BiasFramingBatch  # noqa: E402

SUBTOPIC = "task371-leipzig-drone"
BATCH_SIZE = 8  # Settings.models.bias_framing_batch_size default (tech-lead, 2026-08-12)
PROMPT = Path(__file__).resolve().parent.parent / "newsresearch" / "llm" / "prompts" / "bias_framing.txt"


def load_clusters(pool) -> dict[str, list[dict]]:
    """cluster_id -> its asserting rows. Omitting rows are excluded by design."""
    with pool.connection() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT cluster_id FROM claim_clusters WHERE subtopic_id = %s", (SUBTOPIC,)
            ).fetchall()
        ]
    ids.sort(key=lambda c: int(c.split(":")[1]))
    out = {}
    for cid in ids:
        rows = [r for r in read_cluster_article_relations(pool, cid) if r["relation"] == "asserts"]
        if rows:
            out[cid] = rows
    return out


def format_batch(batch: dict[str, list[dict]]) -> str:
    """Render one batch of clusters into the prompt's {clusters} block."""
    parts = []
    for cid, rows in batch.items():
        # The distinct-article roster is supplied, never counted by the model:
        # iteration 2/3 systematically mistook "one domain" for "one article".
        # dict.fromkeys keeps first-seen order stable.
        distinct = list(dict.fromkeys(r["article_id"] for r in rows))
        lines = [
            f"Cluster {cid}",
            f"Distinct articles in this cluster ({len(distinct)}): " + ", ".join(distinct),
            "Claims (article id | source domain | claim):",
        ]
        lines += [f"- {r['article_id']} | {r['domain']} | {r['claim_text']}" for r in rows]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def run(repeat: int, out_path: Path) -> None:
    settings = Settings()
    pool = init_db(settings.database_url)
    clusters = load_clusters(pool)
    print(f"{len(clusters)} clusters with asserting claims in {SUBTOPIC}")

    chain = ChatPromptTemplate.from_template(PROMPT.read_text()) | get_chat_model(
        "bias_framing"
    ).with_structured_output(BiasFramingBatch)

    ids = list(clusters)
    batches = [
        {c: clusters[c] for c in ids[i : i + BATCH_SIZE]} for i in range(0, len(ids), BATCH_SIZE)
    ]

    results: list[dict] = []
    for run_i in range(repeat):
        for b_i, batch in enumerate(batches):
            res = chain.invoke({"clusters": format_batch(batch)})
            print(f"run {run_i} batch {b_i}: asked {len(batch)}, got {len(res.clusters)}")
            results.append(
                {"run": run_i, "batch": b_i, "asked": list(batch), "output": res.model_dump()}
            )

    payload = {
        "subtopic": SUBTOPIC,
        "batch_size": BATCH_SIZE,
        "model_stage": "bias_framing",
        "model": settings.models.bias_framing,
        "input_clusters": {
            cid: [
                {"article_id": r["article_id"], "domain": r["domain"], "claim_text": r["claim_text"]}
                for r in rows
            ]
            for cid, rows in clusters.items()
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_path}")


def check(out_path: Path) -> None:
    """Mechanical checks over the saved output. Not a quality eval -- see the write-up."""
    payload = json.loads(out_path.read_text())
    inputs = payload["input_clusters"]
    banned = (
        "left-lean",
        "left lean",
        "right-lean",
        "right wing",
        "left wing",
        "centrist",
        "liberal",
        "conservative",
        "progressive",
        "partisan",
        "pro-russia",
        "anti-russia",
        "ideolog",
    )
    counting = ("two sources", "three sources", "four sources", "both outlets", "most sources", "all sources")
    fails: list[str] = []

    for r in payload["results"]:
        asked = set(r["asked"])
        got = [c["cluster_id"] for c in r["output"]["clusters"]]
        if set(got) != asked:
            fails.append(f"run{r['run']} batch{r['batch']}: cluster_id mismatch {sorted(set(got) ^ asked)}")

        for c in r["output"]["clusters"]:
            cid = c["cluster_id"]
            src = inputs.get(cid, [])
            valid_ids = {x["article_id"] for x in src}
            listed = [a["article_id"] for a in c["per_article"]]

            if set(listed) != valid_ids:
                fails.append(f"{cid}: per_article ids {sorted(set(listed) ^ valid_ids)} wrong")
            if len(listed) != len(set(listed)):
                fails.append(f"{cid}: duplicate per_article entries")

            text = " ".join([c["shared_focus"], c["divergence_note"]]).lower()
            for a in c["per_article"]:
                text += " " + a["contextual_frame"].lower()
                # evidence_quote must be verbatim from THAT article's claims here
                own = " ".join(x["claim_text"] for x in src if x["article_id"] == a["article_id"])
                q = a["evidence_quote"].strip().strip('"').strip()
                if q and q.lower() not in own.lower():
                    fails.append(f"{cid}/{a['article_id']}: quote not verbatim: {q!r}")
                if a["domain"] not in {x["domain"] for x in src if x["article_id"] == a["article_id"]}:
                    fails.append(f"{cid}/{a['article_id']}: wrong domain {a['domain']}")

            for b in banned:
                if b in text:
                    fails.append(f"{cid}: political-lean language {b!r}")
            for b in counting:
                if b in text:
                    fails.append(f"{cid}: source counting {b!r}")
            # Both directions: iteration 2 got the first right but systematically
            # failed the second, counting domains instead of article ids.
            if len(valid_ids) == 1 and c["divergence"] != "single_article_only":
                fails.append(f"{cid}: 1 article but divergence={c['divergence']}")
            if len(valid_ids) > 1 and c["divergence"] == "single_article_only":
                fails.append(
                    f"{cid}: {len(valid_ids)} articles "
                    f"({len({x['domain'] for x in src})} domains) but single_article_only"
                )

    print(f"\n{len(fails)} mechanical failures")
    for f in fails:
        print("  FAIL", f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "phase4_bias_framing_samples.json")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if not a.check_only:
        run(a.repeat, a.out)
    check(a.out)
