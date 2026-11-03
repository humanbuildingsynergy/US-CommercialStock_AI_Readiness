"""Readiness tiers on the 76-cell canon. This is this study's definition, not the readiness
framework's (the framework names no tiers).

Coverage of an intent = graded share of the canon's cells for that intent:
    cov_X(I) = sum over domains of achieved level on (domain, X) / sum of their caps.
A building reaches tier X (status, summary, diagnosis) when cov_X >= THRESHOLD.

The central estimate is discounted per intent exactly as C(I) is discounted (``atlas.declared``):
    cov_X,central = cov_X,floor + (cov_X,declared - cov_X,floor) * r.
``cause_separating`` marks a class set that reaches level 3 (cause separation) on at least one
diagnosis cell, which no class set the atlas assigns reaches (reported as a fact, not a tier).
"""
from __future__ import annotations

from functools import lru_cache

from atlas import canon

TIERS = ("status", "summary", "diagnosis")
THRESHOLD = 0.5
THRESHOLD_SENSITIVITY = (0.4, 0.6)


@lru_cache(maxsize=None)
def coverage(streams: frozenset, intent: str, scoring: str = "primary") -> float:
    c = canon.CANONS["v2"]
    cells = [cell for cell in c.rungs if cell[1] == intent]
    if scoring == "domain_equal_tiers":   # tier variant: each domain's question type weighted equally in the intent
        return sum(canon.level(*cell, set(streams)) / len(c.rungs[cell]) for cell in cells) / len(cells)
    got = sum(canon.level(*cell, set(streams), scoring) for cell in cells)
    return got / sum(len(c.rungs[cell]) for cell in cells)


@lru_cache(maxsize=None)
def cause_separating(streams: frozenset) -> bool:
    c = canon.CANONS["v2"]
    return any(c.achieved_level(d, i, set(streams)) >= 3 for (d, i) in c.rungs if i == "diagnosis")
