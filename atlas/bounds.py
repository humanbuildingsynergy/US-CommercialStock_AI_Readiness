"""Bounds and decompositions of the deterministic map.

(a) Strict census bound: remove every stream class that fewer than 10 % of the census's 45 real
    automated-building graphs declare, from the floor and the declared level alike (this adds the
    whole-building meter, 1/45 declared and 0 linked, to the classes the census-supported reading screens out).
(b) What the number is made of: an exact Shapley decomposition of floor-area-weighted mean central
    C(I) into four stream-class groups (thermostat; metering; BAS HVAC points; occupancy, lighting
    and CO2). For a record, group g contributes phi_g(floor) + r * (phi_g(declared) - phi_g(floor)),
    where phi is the Shapley value of C(I) over the groups; contributions sum to the record's central
    C(I) exactly.
(c) The metering lever (H5) restricted to the floor area §8.4.3 covers (>= 25,000 ft2), with and
    without the HVAC-monitoring -> equipment-power mapping.
(d) The full sensitivity table (frozen and enriched variants) as Supplementary Table S6.

    python -m atlas.bounds      -> results/stage4_bounds/*.{md,csv}, results/supplement/S6_sensitivities.md
    (the deterministic map of an earlier design; these folders are not part of the release results)
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from atlas import canon, cbecs, config, declared, tiers
from atlas.replicates import REPLICATES

Z = 1.96
GROUPS = {
    "thermostat (T, Sp, SpC)": frozenset({"T", "Sp", "SpC"}),
    "metering (Whole, Sub, Sub2, Pwr)": frozenset({"Whole", "Sub", "Sub2", "Pwr"}),
    "BAS HVAC points (SAT, OAT, OAF, Run, Load, ActC, ActP)": frozenset({"SAT", "OAT", "OAF", "Run", "Load", "ActC", "ActP"}),
    "occupancy, lighting, CO2 (Occ, Lgt, CO2)": frozenset({"Occ", "Lgt", "CO2"}),
}
BUNDLE = frozenset({"Lgt", "Pwr", "Sub", "Sub2"})
BUNDLE_NO_PWR = frozenset({"Lgt", "Sub", "Sub2"})


def _fs(s: str) -> frozenset:
    return frozenset() if s == "-" else frozenset(s.split("+"))


def rare_classes() -> frozenset:
    share = declared.census_share()
    vocab = set(canon.STREAMS)
    return frozenset(k for k in vocab if share.get(k, 0.0) < declared.CENSUS_MIN)


# ── (b) Shapley decomposition by stream-class group ─────────────────────────────────────────────
@lru_cache(maxsize=None)
def _phi(streams: frozenset, scoring: str = "primary") -> tuple[float, ...]:
    names = list(GROUPS)
    K = len(names)
    v = {S: canon.readiness(streams & frozenset().union(*(GROUPS[n] for n in S)), scoring)
         for k in range(K + 1) for S in itertools.combinations(names, k)}
    phi = []
    for n in names:
        tot = 0.0
        for k in range(K):
            for S in itertools.combinations([m for m in names if m != n], k):
                Sn = tuple(m for m in names if m in S or m == n)
                tot += math.factorial(k) * math.factorial(K - k - 1) / math.factorial(K) * (v[Sn] - v[S])
        phi.append(tot)
    return tuple(phi)
