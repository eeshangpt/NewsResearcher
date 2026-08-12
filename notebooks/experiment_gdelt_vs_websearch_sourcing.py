# %% [markdown]
# # GDELT+RSS vs. DuckDuckGo+Tavily sourcing: side-by-side comparison
#
# GDELT DOC 2.0 is persistently IP-blocked (issue #101) and the Google News
# RSS backfill fallback is also broken (issue #115). This notebook is a real,
# non-mocked comparison of two sourcing strategies for the same topic:
#
# 1. **Current flow**: `agents/sourcing_agent.py` exactly as production calls
#    it (GDELT -> RSS -> Google News backfill-if-thin -> dedup -> reputation
#    score -> threshold filter). No mocking. GDELT is expected to soft-fail;
#    we report what actually happens rather than hiding it.
# 2. **New flow (candidate)**: DuckDuckGo search + Tavily search (same
#    libraries prototyped standalone in `news_agent.py`) for discovery, then
#    this project's real `sourcing/fulltext.py::fetch_fulltext_for_cluster`
#    for full-text extraction (trafilatura, no vendor lock to the prototype's
#    own trafilatura wiring).
#
# Both sets of surviving articles get scored through the real
# `reputation/scorer.py` so the comparison lands on the same quality axis
# Gate 1 already filters on, not just raw article counts.
#
# This is an exploratory design script, not production code. Nothing here is
# wired into the pipeline. `backend-engineer` integrates any recommendation
# that comes out of this behind `Settings`, in its own branch.
#
# Run this as a Jupyter notebook (py:percent format, `# %%` cells) or as a
# plain script (`uv run python notebooks/experiment_gdelt_vs_websearch_sourcing.py`).
#
# **Requirements**: `NEWSRESEARCH_DATABASE_URL` (reputation cache -- Docker
# Postgres must be up), `OPENAI_API_KEY` unused here (no LLM calls in this
# script -- pure sourcing/reputation comparison), `TAVILY_API_KEY` for the
# new flow's Tavily leg. DuckDuckGo search needs no key. If `TAVILY_API_KEY`
# is missing, the Tavily leg is skipped and reported as skipped -- the script
# still runs end to end on DuckDuckGo alone.
#
# No article full text is written to disk anywhere in this script (matches
# the no-full-text-storage rule) -- full text is fetched, measured (length,
# success/fail), and discarded in-memory; only URLs/domains/scores are
# printed or held in local variables.

# %%
from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# ## Topic and run metadata
#
# Pick something with real, current news coverage. State what and when this
# was actually run -- the printed output below is this run's real result,
# not a canned example.

# %%
TOPIC = "India Pakistan ceasefire talks"
KEYWORDS = ["India", "Pakistan", "ceasefire"]
LOOKBACK_DAYS = 2
RUN_TIMESTAMP = time.strftime("%Y-%m-%d %H:%M %Z")
print(f"Topic: {TOPIC!r}")
print(f"Keywords (current-flow GDELT/RSS query): {KEYWORDS!r}")
print(f"Lookback: {LOOKBACK_DAYS} days")
print(f"Run started: {RUN_TIMESTAMP}")

# %% [markdown]
# ## Part 1 -- current flow: real `sourcing_agent` (GDELT + RSS + backfill)
#
# Calls the actual production orchestrator, no mocking. Needs
# `NEWSRESEARCH_DATABASE_URL` for the reputation cache (`docker compose up -d`
# app Postgres). GDELT is expected to soft-fail per issue #101 -- watch the
# printed `gdelt_count` below rather than assuming.

# %%
from psycopg_pool import ConnectionPool  # noqa: E402

from newsresearch.agents.sourcing_agent import sourcing_agent  # noqa: E402
from newsresearch.config import Settings  # noqa: E402
from newsresearch.persistence.db import init_db  # noqa: E402
from newsresearch.sourcing import gdelt, rss  # noqa: E402
from newsresearch.sourcing.gdelt import GDELTError  # noqa: E402

settings = Settings()
if not settings.database_url:
    raise RuntimeError("NEWSRESEARCH_DATABASE_URL not set -- copy .env, start docker compose Postgres.")

# `reputation/signals.py::get_domain_age_years`/`check_https_and_about_page`
# hardcode their own WHOIS/HTTPS timeouts (10s/5s) with no `Settings` knob --
# real, unmocked calls, just slow (observed: 10+ min wall-clock in this
# sandbox's network, largely WHOIS/DNS timeouts against unknown-tier
# international domains a broad GDELT hit pulls in). Monkeypatched here,
# notebook-local only, to a shorter timeout purely so this script finishes in
# a reasonable interactive session -- still real network calls, same
# soft-fail behavior on timeout, just less patient. Not a production change;
# remove this patch to reproduce production's exact timeout behavior.
# ponytail: notebook-local timeout patch trades a little signal-collection
# patience for a runnable notebook; do not port this into signals.py itself.
import functools  # noqa: E402

from newsresearch.reputation import signals as _signals_module  # noqa: E402

_signals_module.get_domain_age_years = functools.partial(_signals_module.get_domain_age_years, timeout=4)
_signals_module.check_https_and_about_page = functools.partial(_signals_module.check_https_and_about_page, timeout=2.5)

# Issue #101: GDELT is real-429 rate-limited (not a hard IP block in this
# run -- it returns partial, capped-per-window results after retrying), and
# `gdelt_max_retries=5` with exponential backoff up to 40s/attempt across a
# multi-day, auto-split lookback window can take 10+ minutes wall-clock for
# a single call. Retries capped to 2/short backoff here ONLY so this
# exploratory notebook finishes in a reasonable time -- still a real,
# unmocked GDELT call, just less patient than production's default. Change
# back to `Settings()` defaults if you want the full production retry
# behavior reproduced exactly (and are willing to wait for it).
# ponytail: shortened retry budget trades a little probe realism for a
# runnable notebook; revert to defaults for a production-fidelity re-check.
fast_probe_settings = settings.model_copy(deep=True)
fast_probe_settings.sourcing.gdelt_max_retries = 2
fast_probe_settings.sourcing.gdelt_backoff_base_seconds = 3.0

pool: ConnectionPool = init_db(settings.database_url)

# Probe GDELT directly first (separately from sourcing_agent's own soft-fail
# swallow) purely to illustrate GDELT's *current* real failure mode (issue
# #101) descriptively -- this probe uses the shortened retry budget above so
# it doesn't itself take 10+ minutes, so its pass/fail is NOT the source of
# truth for whether GDELT actually contributed to current_flow below (that
# needs production's real retry budget, which sourcing_agent() uses next).
try:
    gdelt_probe = gdelt.fetch(KEYWORDS, LOOKBACK_DAYS, settings=fast_probe_settings)
    gdelt_status = f"succeeded, {len(gdelt_probe)} article(s)"
except GDELTError as exc:
    gdelt_probe = []
    gdelt_status = f"GDELTError (soft-failed, as expected per issue #101): {exc}"

print(f"GDELT direct probe (shortened retry budget, descriptive only): {gdelt_status}")

# The authoritative "did GDELT actually contribute to current_flow" signal:
# sourcing_agent() below makes its own internal, full-retry-budget GDELT call
# and logs exactly one WARNING ("gdelt.fetch failed; continuing with RSS...")
# on soft-fail -- watch for that real log line rather than reusing the
# probe's separate (shortened-retry, so not equivalent) result.
import logging as _logging  # noqa: E402


class _GDELTSoftFailWatcher(_logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.saw_soft_fail = False

    def emit(self, record: _logging.LogRecord) -> None:
        if "gdelt.fetch failed" in record.getMessage():
            self.saw_soft_fail = True


_gdelt_watcher = _GDELTSoftFailWatcher()
_logging.getLogger("newsresearch.agents.sourcing_agent").addHandler(_gdelt_watcher)

current_flow_results = sourcing_agent(KEYWORDS, LOOKBACK_DAYS, pool=pool, settings=settings)

print(f"\ncurrent_flow: {len(current_flow_results)} article(s) survived dedup + reputation threshold")
for scored in current_flow_results[:10]:
    a = scored.article
    print(f"  {a['domain']:30s} score={scored.reputation_score:.2f} tier={scored.reputation_tier:8s} {a['url']}")
if len(current_flow_results) > 10:
    print(f"  ... and {len(current_flow_results) - 10} more")

# %% [markdown]
# ## Part 2 -- new flow: DuckDuckGo + Tavily discovery, real fulltext fetch
#
# Same discovery libraries as `news_agent.py`'s researcher tools
# (`DuckDuckGoSearchResults`, `TavilySearch`), but only the discovery+fetch
# slice -- no LangGraph researcher/writer/critic loop, no LLM calls at all.
# Full text comes from this project's real
# `sourcing/fulltext.py::fetch_fulltext_for_cluster`, not the prototype's
# inline trafilatura call.

# %%
from langchain_community.tools import DuckDuckGoSearchResults  # noqa: E402

from newsresearch.sourcing.fulltext import fetch_fulltext_for_cluster  # noqa: E402

ddg_tool = DuckDuckGoSearchResults(output_format="list", max_results=15)
ddg_raw = ddg_tool.invoke(TOPIC)
# output_format="list" -> list[dict] with "link"/"title"/"snippet" keys.
print(f"DuckDuckGo: {len(ddg_raw)} raw result(s)")

tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
tavily_raw: list[dict] = []
if tavily_key:
    from langchain_tavily import TavilySearch  # noqa: E402

    tavily_tool = TavilySearch(max_results=15, topic="news")
    tavily_response = tavily_tool.invoke({"query": TOPIC})
    tavily_raw = tavily_response.get("results", []) if isinstance(tavily_response, dict) else []
    print(f"Tavily: {len(tavily_raw)} raw result(s)")
else:
    print("Tavily: SKIPPED -- TAVILY_API_KEY not set. New-flow numbers below are DuckDuckGo-only.")


def _domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


# Normalize both sources to the article dict shape the rest of this project
# uses ({"url", "title", "domain", "source_type"}), then de-dup by URL --
# same intent as sourcing/dedup.py::dedup_by_url, kept inline since this is
# a two-source, notebook-local merge, not the production dedup pass.
new_flow_candidates: dict[str, dict] = {}
for r in ddg_raw:
    url = r.get("link") or r.get("url")
    if not url:
        continue
    new_flow_candidates.setdefault(url, {"url": url, "title": r.get("title", ""), "domain": _domain_of(url), "source_type": "duckduckgo"})
for r in tavily_raw:
    url = r.get("url")
    if not url:
        continue
    new_flow_candidates.setdefault(url, {"url": url, "title": r.get("title", ""), "domain": _domain_of(url), "source_type": "tavily"})

new_flow_articles = list(new_flow_candidates.values())
print(f"\nnew_flow: {len(new_flow_articles)} distinct URL(s) after cross-source URL dedup (DuckDuckGo + Tavily)")

# %% [markdown]
# ### Full-text fetch spot-check (real fetcher, in-memory only)
#
# Every new-flow candidate gets a real `fetch_fulltext_for_cluster` call.
# This reports the fetch *success rate* -- a low rate here would mean the
# new flow's discovery URLs are often paywalled/unfetchable even though
# search found them, which matters as much as raw discovery count.

# %%
fulltext_results = fetch_fulltext_for_cluster(new_flow_articles, settings=settings)
fulltext_by_url = {r["url"]: r["fulltext"] for r in fulltext_results}
fetch_ok = sum(1 for v in fulltext_by_url.values() if v)
fetch_total = len(fulltext_by_url)
print(f"new_flow fulltext fetch: {fetch_ok}/{fetch_total} succeeded ({fetch_ok / fetch_total:.0%})" if fetch_total else "new_flow: no candidates to fetch")

# require required fields (url/title/domain) same gate sourcing_agent applies,
# so new_flow isn't compared against current_flow on a looser fields bar.
new_flow_valid = [a for a in new_flow_articles if a.get("url") and a.get("title") and a.get("domain")]
print(f"new_flow: {len(new_flow_valid)}/{len(new_flow_articles)} article(s) have required url/title/domain fields")

# %% [markdown]
# ## Part 3 -- score new-flow articles through the real reputation scorer
#
# Same `reputation/scorer.py::score_domain` + `reputation/signals.py` used by
# `sourcing_agent` for Part 1, so both flows land on the same 0..1 scale Gate
# 1 filters on. Presence-frequency here is new-flow-batch-relative (DuckDuckGo
# vs. Tavily source_type), matching the batch-relative convention
# `scorer.py`'s own docstring documents (not directly comparable in absolute
# terms to current-flow's GDELT/RSS/backfill-relative presence-frequency --
# both are internally consistent, not cross-normalized, flagged here rather
# than silently treated as apples-to-apples on that one sub-signal).

# %%
from newsresearch.reputation import cache, signals  # noqa: E402
from newsresearch.reputation import scorer as reputation_scorer  # noqa: E402


def score_articles(articles: list[dict], pool: ConnectionPool, settings: Settings) -> list[dict]:
    presence = signals.get_presence_frequency_scores(articles)
    scored = []
    for article in articles:
        domain = article["domain"]
        cached = cache.get_fresh_domain_reputation(pool, domain, settings=settings)
        if cached is not None:
            rep = cached
        else:
            rep = reputation_scorer.score_domain(
                domain,
                domain_age_years=signals.get_domain_age_years(domain),
                backlink_proxy=signals.get_backlink_proxy_score(domain),
                presence_frequency=presence.get(domain),
                https_present=signals.check_https_and_about_page(domain)["https_present"],
                about_page_present=signals.check_https_and_about_page(domain)["about_page_present"],
                settings=settings,
            )
            cache.write_domain_reputation(pool, rep)
        scored.append({**article, "reputation_score": rep["final_score"], "reputation_tier": rep["tier"]})
    return scored


new_flow_scored = score_articles(new_flow_valid, pool, settings)
new_flow_pass = [a for a in new_flow_scored if a["reputation_score"] >= settings.reputation.min_score_threshold]
print(f"new_flow: {len(new_flow_pass)}/{len(new_flow_scored)} article(s) meet min_score_threshold={settings.reputation.min_score_threshold:.2f}")
for a in sorted(new_flow_scored, key=lambda x: -x["reputation_score"])[:10]:
    flag = "PASS" if a["reputation_score"] >= settings.reputation.min_score_threshold else "fail"
    print(f"  [{flag}] {a['domain']:30s} score={a['reputation_score']:.2f} tier={a['reputation_tier']:8s} {a['url']}")

pool.close()

# %% [markdown]
# ## Part 4 -- comparison
#
# Not just two counts side by side: real domain overlap/distinctness, and
# both sets on the same reputation-threshold bar.

# %%
current_domains = {s.article["domain"] for s in current_flow_results}
new_domains_all = {a["domain"] for a in new_flow_valid}
new_domains_pass = {a["domain"] for a in new_flow_pass}

overlap = current_domains & new_domains_all
current_only = current_domains - new_domains_all
new_only = new_domains_all - current_domains

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"current_flow (GDELT+RSS+backfill): {len(current_flow_results)} article(s), {len(current_domains)} distinct domain(s)")
print(f"  GDELT contributed (sourcing_agent's own real, full-retry-budget call): {'no -- soft-failed, RSS(+backfill)-only' if _gdelt_watcher.saw_soft_fail else 'yes'}")
print(f"new_flow (DuckDuckGo+Tavily), all discovered: {len(new_flow_valid)} article(s), {len(new_domains_all)} distinct domain(s)")
print(f"new_flow, passing reputation threshold: {len(new_flow_pass)} article(s), {len(new_domains_pass)} distinct domain(s)")
print(f"new_flow fulltext fetch success rate: {fetch_ok}/{fetch_total} ({fetch_ok / fetch_total:.0%})" if fetch_total else "new_flow fulltext: n/a")
print()
print(f"Domain overlap (in both flows): {len(overlap)} -- {sorted(overlap)}")
print(f"current_flow-only domains: {len(current_only)} -- {sorted(current_only)}")
print(f"new_flow-only domains: {len(new_only)} -- {sorted(new_only)}")
print()
if not tavily_key:
    print("LIMITATION: Tavily leg skipped (no TAVILY_API_KEY) -- new_flow numbers above are DuckDuckGo-only, understating new_flow's real reach.")
if fetch_total and fetch_ok / fetch_total < 0.7:
    print(f"LIMITATION: new_flow fulltext fetch succeeded on only {fetch_ok / fetch_total:.0%} of discovered URLs -- discovery count alone overstates usable new_flow yield.")
print()
print("This is a single-topic, single-run spot-check, not a measured eval across")
print("a fixture/golden set (PRD/TRD Evaluation Strategy §7 gap -- v1 has no")
print("automated golden-dataset evaluation). Re-run on a few more topics before")
print("treating any count/overlap above as a stable conclusion rather than one")
print("data point.")
print()
print("LIMITATION (observed this run): GDELT is real-429 rate-limited rather than")
print("a hard IP block right now (issue #101) -- it returns real, partial (per-window")
print("250-record-capped) results after retrying, not a clean failure. When GDELT")
print("does return international hits, most of their domains are 'unknown' tier and")
print("need a live WHOIS + HTTPS/about-page check each (Part 1's reputation scoring)")
print("-- each one soft-fails correctly on timeout/DNS failure (no crash, neutral")
print("0.5 fallback per NFR-3) but sequential real network calls across dozens of")
print("unknown domains made a full run take 10+ minutes wall-clock in this sandbox.")
print("Narrow KEYWORDS/LOOKBACK_DAYS if you need a faster iteration loop.")

# %%
if __name__ == "__main__":
    pass
