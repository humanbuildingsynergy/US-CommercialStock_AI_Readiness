"""The code-clause table (edition -> clauses -> stream classes) and the code-required level per record.

The code-required level of a record is C(I) on the readiness framework's 76-level canon (``atlas.canon``, copied) over
the stream classes the code-clause table (``tables/C_clauses.csv``) mandates for it. A row applies when:
  - its ``reading`` belongs to the selected reading (READINGS below),
  - the record's edition rung for the row's system is at least ``rung_from`` — the construction
    rung for ``new`` rows, the post-renovation rung for ``new+alteration`` rows, and for
    ``alteration`` rows only when that system was renovated (the edition table),
  - the record meets the row's ``condition`` (CONDITIONS below; every condition is a CBECS test).
Three levels per record and reading: low / central / high, from the edition interval.
"""
from __future__ import annotations

import csv
from functools import lru_cache

import pandas as pd

from atlas import canon, cbecs, config, vintage

LEVELS = ("low", "central", "high")

# Each reading is a set of the table's ``reading`` tags. "main" follows the clause text (the primary result);
# "early_occ" is the sensitivity that credits occupancy sensing from 90.1-2004 and 2010, and "alt_lighting" the one
# that credits it from a 2016-rung lighting alteration (C06, S24). Rows read "not_used" enter no reading.
READINGS: dict[str, frozenset[str]] = {
    "main": frozenset({"core", "occ_late", "meter_clause"}),
    "early_occ": frozenset({"core", "occ_early", "occ_late", "meter_clause"}),
    "alt_lighting": frozenset({"core", "occ_late", "occ_alt", "meter_clause"}),
}
# C06 (reading occ_alt) is held in the multiple imputation of the main and early_occ readings (atlas.mi.make_imp), so
# its compliance draws keep their place in the random stream; atlas.mi.evaluate applies it only when Cfg.alt_lighting
# is set. Under exception 2 a lamp/ballast or one-for-one retrofit complies by scheduled shutoff, and the survey's
# lighting-upgrade item cannot separate those retrofits from other alterations.
HELD: dict[str, frozenset[str]] = {"main": frozenset({"occ_alt"}), "early_occ": frozenset({"occ_alt"})}
# 90.1-2013 §8.4.3 exception f (2010 addendum bn): hotels, motels and restaurants. CBECS PBAPLUS separates them:
# 32 fast food, 33 restaurant/cafeteria, 38 hotel/resort, 39 motel/inn/bed and breakfast.
EXEMPT_843 = (32, 33, 38, 39)
# Clause rows whose compliance draw takes its uniforms from a stream of its own (atlas.mi.evaluate, atlas.draws.sample),
# so adding them moves no other draw: C15, 90.1-2013 §10.4.5.
OWN_STREAM = frozenset({"C15"})
READING_NOTES = {
    "main": "occupancy from 90.1-2013 (Table 9.6.1 automatic OFF on departure); §8.4.3 monitoring from 2013 for new "
            "buildings >= 25,000 ft2 other than hotels, motels and restaurants; §10.4.5 district-energy monitoring "
            "from 2013 for new buildings >= 25,000 ft2 that use district chilled water, steam or hot water",
    "alt_lighting": "as main, plus occupancy from a 2016-rung lighting alteration (C06; sensitivity S24)",
    "early_occ": "as main, plus occupancy from 2004 (> 5,000 ft2) and 2010 (all sizes), where a "
                 "timer switch also complies",
}

CONDITIONS = {
    "conditioned": lambda d: (d.HEATP.fillna(0) > 0) | d.COOLP.notna(),
    "electric": lambda d: d.ELUSED == 1,
    "lit": lambda d: d.OCSN.notna(),       # CBECS's lighting-controls universe: lit, with electricity
    "sqft>5000": lambda d: d.SQFT > 5000,
    "sqft>=25000": lambda d: d.SQFT >= 25000,
    "not-hotel-motel-restaurant": lambda d: ~d.PBAPLUS.isin(EXEMPT_843),
    "district": lambda d: (d.CHWT == 1) | (d.STHW == 1),   # the survey items of reported control E13
}
CONDITION_NOTES = {
    "conditioned": "HEATP > 0 or COOLP present (the EMCS universe)",
    "electric": "ELUSED = 1",
    "lit": "asked the lighting-controls block (lit, with electricity)",
    "sqft>5000": "SQFT > 5,000 (no longer used: the threshold belongs to §9.4.1.1 automatic shutoff, not §9.4.1.2)",
    "sqft>=25000": "SQFT >= 25,000 (§8.4.3 and §10.4.5 exempt buildings below 25,000 ft2)",
    "not-hotel-motel-restaurant": "PBAPLUS not 32, 33, 38 or 39 (§8.4.3 exception f: hotels, motels and restaurants)",
    "district": "CHWT = 1 or STHW = 1 (district chilled water, steam or hot water; §10.4.5)",
}


@lru_cache(maxsize=1)
def table_c() -> list[dict[str, str]]:
    with open(config.TABLES / "C_clauses.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _condition(df: pd.DataFrame, expr: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for term in expr.split("&"):
        if term not in CONDITIONS:      # "(unobservable)" conditions never hold
            return pd.Series(False, index=df.index)
        mask &= CONDITIONS[term](df)
    return mask


def _renovated(df: pd.DataFrame, system: str) -> pd.Series:
    vars_ = [v for v, systems in vintage.renovation_rules().items() if system in systems]
    return (df.RENOV == 1) & df[vars_].eq(1).any(axis=1)


def streams(df: pd.DataFrame, rungs: pd.DataFrame, reading: str, level: str) -> pd.Series:
    """Per record: the frozenset of stream classes the code-clause table mandates under *reading* at *level*."""
    L = vintage.ladder()
    tags = READINGS[reading]
    have = {i: set() for i in df.index}
    for row in table_c():
        if row["reading"] not in tags:
            continue
        need = L.index(row["rung_from"])
        sys_ = row["system"]
        if row["path"] == "new":
            ok = rungs[f"new_{level}"] >= need
        elif row["path"] == "new+alteration":
            ok = rungs[f"{sys_}_{level}"] >= need
        else:  # alteration
            ok = _renovated(df, sys_) & (rungs[f"{sys_}_{level}"] >= need)
        ok &= _condition(df, row["condition"])
        for i in df.index[ok]:
            have[i] |= set(row["streams"].split("+"))
    return pd.Series({i: frozenset(s) for i, s in have.items()})


@lru_cache(maxsize=None)
def _score(s: frozenset) -> float:
    return canon.ceiling(s, "v2").score


def floors(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = cbecs.load() if df is None else df
    rungs = vintage.assign(df)
    cols = {}
    for reading in READINGS:
        for level in LEVELS:
            s = streams(df, rungs, reading, level)
            cols[f"{reading}_{level}_streams"] = s.map(lambda x: "+".join(sorted(x)) or "-")
            cols[f"{reading}_{level}"] = s.map(_score)
    return pd.DataFrame(cols, index=df.index)


SIZE_CLASSES = ("<= 5,000", "5,001-24,999", ">= 25,000")


def size_class(sqft: pd.Series) -> pd.Series:
    return pd.cut(sqft, [0, 5000, 24999, float("inf")], labels=SIZE_CLASSES).astype(str)
