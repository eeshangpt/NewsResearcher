"""Reputation scoring formula (TRD 4.2, Story 1.7, Task 1.7.1).

Combines the four `reputation/signals.py` collectors (Stories 1.5/1.6) into a
single per-domain reputation score, per TRD 4.2:

    score(domain) = base_tier_score(domain)
                    + clip(w1 * norm(domain_age_years)
                           + w2 * norm(backlink_proxy)
                           + w3 * norm(presence_frequency)
                           + w4 * legitimacy_flags(https_present, about_page_present),
                           -adjustment_bound, +adjustment_bound)

`base_tier_score` comes from `data/trusted_outlets.yaml`'s tier (`wire`/
`major`), via `Settings.reputation.base_score_{wire,major}`, or
`Settings.reputation.base_score_unknown` for a domain absent from that
whitelist. `w1..w4` and `adjustment_bound` are `Settings.reputation` fields
(TRD 4.2: "tunable weights summing to a bounded adjustment range, e.g.
±0.3"). The summed adjustment is explicitly clipped to `[-adjustment_bound,
+adjustment_bound]` *after* summing -- not just relied on via weight
configuration staying in range -- so a misconfigured weight override, or any
individual normalized signal drifting outside `[0, 1]`, can never push a
domain's score outside the configured bound (EXECUTION_PLAN Task 1.7.1:
"clip, don't sum unbounded").

Each of the four raw signals is soft-fail/neutral-tolerant on the way in,
matching `reputation/signals.py`'s own fail-open convention: a `None`
domain_age/presence_frequency, or a `None` https/about-page flag, normalizes
to `0.5` (neutral) rather than being treated as a penalty.

--------------------------------------------------------------------------
Required design decision -- presence-frequency denominator (EXECUTION_PLAN
Task 1.7.1 integration-checkpoint note)
--------------------------------------------------------------------------
`signals.get_presence_frequency_scores` normalizes each domain's distinct
source-type count against *that batch's own* distinct-source-type count (1,
2, or 3, depending on whether GDELT/RSS/Google-News-backfill all fired for
that particular fetch), not a fixed 3-source-type universe. Since
`domain_reputation` is a cross-run *persisted* cache, the same domain's
cached `w3` contribution can shift purely because backfill happened to fire
(or not) in whichever run triggered its staleness-based recompute --
unrelated to the domain's actual legitimacy.

Chosen: KEEP the batch-relative denominator as `signals.py` produces it (do
not fix it to a 3-source-type universe here). Reasoning:

1. The two places a fixed-denominator fix could live without duplicating
   `get_presence_frequency_scores`'s counting logic in a second location --
   `reputation/signals.py` itself, or the call site in
   `agents/sourcing_agent.py` that supplies the raw fetch batch -- are both
   out of this task's scope (Story 1.7 owns `scorer.py`/`cache.py` only).
   Reimplementing a second, differently-denominated presence-frequency
   counter inside `scorer.py` would create two divergent definitions of the
   same signal, which is a worse outcome than the noise being accepted here.
2. The magnitude at risk is small and bounded: `weight_presence_frequency`
   defaults to 0.05 out of a ±0.3 `adjustment_bound` -- the worst-case swing
   for a given domain (batch-relative 1.0 when only GDELT+RSS ran, vs. 0.67
   for the same domain once backfill also fires) moves the final score by at
   most `0.05 * (1.0 - 0.67) ≈ 0.017`, comparable to or smaller than the
   noise already inherent in WHOIS-age drift or a Tranco-snapshot refresh
   between recomputes.
3. It self-corrects: `Settings.reputation.staleness_days` (default 30) bounds
   how long any single noisy recompute's cached score persists -- it is not
   a permanent mis-score, just a bounded-duration one.

If Task 1.11.1's empirical good-vs-bad spot-check later shows this noise
actually flips a domain across `min_score_threshold`, the fix belongs in
`reputation/signals.py::get_presence_frequency_scores`'s denominator (change
to a fixed 3), filed as a follow-up against Task 1.5.1/1.6.x specifically --
not against this scorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml

from newsresearch.config import Settings

_TRUSTED_OUTLETS_PATH = Path(__file__).resolve().parent.parent / "data" / "trusted_outlets.yaml"

# A domain this old or older gets full (1.0) domain-age credit. 20 years
# comfortably covers "clearly long-established" (TRD 4.2 only needs enough
# resolution to distinguish that from "not clearly established," per the
# Tranco-signal precedent in reputation/signals.py), without needing extreme
# ages (50+ years) to saturate the signal.
_MAX_DOMAIN_AGE_YEARS_FOR_FULL_SCORE = 20.0

# Neutral value applied when a signal is unavailable/unknown -- matches the
# fail-open convention already used by every Story 1.6 collector (Tranco's
# absent-domain 0.5, HTTPS/about-page's unreachable-domain None) so this
# scorer doesn't need a different penalty-vs-neutral rule per signal.
_NEUTRAL_SIGNAL_VALUE = 0.5

_trusted_outlets_cache: dict[str, str] | None = None


def _load_trusted_outlets() -> dict[str, str]:
    with _TRUSTED_OUTLETS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_trusted_outlets() -> dict[str, str]:
    global _trusted_outlets_cache
    if _trusted_outlets_cache is None:
        _trusted_outlets_cache = _load_trusted_outlets()
    return _trusted_outlets_cache


def get_base_tier_score(domain: str, settings: Settings | None = None) -> tuple[float, str]:
    """`(base_score, tier)` for `domain` via `data/trusted_outlets.yaml`.

    `tier` is `"wire"`, `"major"`, or `"unknown"` (domain absent from the
    whitelist, or present under a tier string this scorer doesn't recognize).
    """
    settings = settings if settings is not None else Settings()
    tier = _get_trusted_outlets().get(domain.strip().lower())

    if tier == "wire":
        return settings.reputation.base_score_wire, "wire"
    if tier == "major":
        return settings.reputation.base_score_major, "major"
    return settings.reputation.base_score_unknown, "unknown"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_domain_age(domain_age_years: float | None) -> float:
    if domain_age_years is None:
        return _NEUTRAL_SIGNAL_VALUE
    return _clip(domain_age_years / _MAX_DOMAIN_AGE_YEARS_FOR_FULL_SCORE, 0.0, 1.0)


def _normalize_backlink_proxy(backlink_proxy: float) -> float:
    # `signals.get_backlink_proxy_score` already returns a 0..1 value (with
    # its own 0.5-neutral absent-domain handling) -- clip defensively here
    # too so a caller passing a raw/unclipped value can't defeat the bound.
    return _clip(backlink_proxy, 0.0, 1.0)


def _normalize_presence_frequency(presence_frequency: float | None) -> float:
    if presence_frequency is None:
        return _NEUTRAL_SIGNAL_VALUE
    return _clip(presence_frequency, 0.0, 1.0)


def _legitimacy_flag_value(flag: bool | None) -> float:
    if flag is None:
        return _NEUTRAL_SIGNAL_VALUE
    return 1.0 if flag else 0.0


def _legitimacy_flags_score(https_present: bool | None, about_page_present: bool | None) -> float:
    return (_legitimacy_flag_value(https_present) + _legitimacy_flag_value(about_page_present)) / 2.0


class DomainReputationScore(TypedDict):
    domain: str
    tier: str
    base_score: float
    heuristic_adjustment: float
    final_score: float


def score_domain(
    domain: str,
    domain_age_years: float | None,
    backlink_proxy: float,
    presence_frequency: float | None,
    https_present: bool | None,
    about_page_present: bool | None,
    settings: Settings | None = None,
) -> DomainReputationScore:
    """TRD 4.2's reputation-scoring formula for a single domain.

    Every signal argument is the already-collected raw output of the
    corresponding `reputation/signals.py` collector (or `None`/neutral where
    that collector soft-failed) -- this function only implements the
    weighting/combination formula, it does not call the collectors itself.
    See the module docstring for the presence-frequency denominator design
    decision and the clip-not-sum-unbounded requirement.
    """
    settings = settings if settings is not None else Settings()
    rep = settings.reputation

    base_score, tier = get_base_tier_score(domain, settings=settings)

    norm_age = _normalize_domain_age(domain_age_years)
    norm_backlink = _normalize_backlink_proxy(backlink_proxy)
    norm_presence = _normalize_presence_frequency(presence_frequency)
    legitimacy = _legitimacy_flags_score(https_present, about_page_present)

    raw_adjustment = (
        rep.weight_domain_age * norm_age
        + rep.weight_backlink_proxy * norm_backlink
        + rep.weight_presence_frequency * norm_presence
        + rep.weight_legitimacy_flags * legitimacy
    )
    adjustment = _clip(raw_adjustment, -rep.adjustment_bound, rep.adjustment_bound)

    return {
        "domain": domain.strip().lower(),
        "tier": tier,
        "base_score": base_score,
        "heuristic_adjustment": adjustment,
        "final_score": base_score + adjustment,
    }
