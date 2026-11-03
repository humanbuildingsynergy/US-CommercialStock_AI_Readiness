"""Table E (reported controls -> declared stream classes), the linked discount, and the
three numbers per record: floor / central / declared, with readiness-tier coverage.

  declared  = C(I) over (floor classes ∪ Table E classes the record's reported controls imply)
  central   = floor + (declared - floor) * r

The same discount is applied per intent for the tiers (``atlas.tiers``). r comes from the readiness
framework's ceiling census (45 real Brick graphs, ``tables/R_reference_set.csv``, copied with citation
from the framework's census results), as defined in ``tables/R_discount.csv``. Levels: low uses the
Table V low floor and r's lower bound, high the high floor and upper bound.

PRIMARY holds the primary configuration. Every alternative is computed beside it as a
sensitivity.

    python -m atlas.declared      -> results/reference_set/table_E.md, results/raw/atlas_records.csv
"""
from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np
import pandas as pd

from atlas import canon, cbecs, config, floors, tiers


# Table E readings (sets of row tags). "primary" = the base rows (tag "wo") + the extended rows, without E02.
E_READINGS = {
    "primary": frozenset({"wo", "extended"}),
    "wo": frozenset({"wo", "wo_bas_occ"}),
    "primary_e02": frozenset({"wo", "wo_bas_occ", "extended"}),
    # exploratory enriched readings (not the primary)
    "enriched": frozenset({"wo", "extended", "enriched", "enriched_spc"}),
    "enriched_no_spc": frozenset({"wo", "extended", "enriched"}),
    "enriched_weak": frozenset({"wo", "extended", "enriched", "enriched_spc", "enriched_weak"}),
    # optimistic scenario (atlas.scoring): the enriched rows plus E19 (a BAS reads equipment power); E20 stays out
    "optimistic": frozenset({"wo", "extended", "enriched", "enriched_spc", "enriched_weak"}),
}
E_READINGS["enriched_flagged"] = frozenset({"wo", "extended", "enriched", "enriched_spc", "enriched_weak"})
# sensitivity S10 on the rows it names (all six flagged rows, E07 E12 E13 E17 E18 E20, without E19)
E_EXCLUDE = {"optimistic": frozenset({"E20"}), "enriched_flagged": frozenset({"E19"})}
PRIMARY = {"floor": "main", "controls": "primary", "r": "R1"}
PRE_REGISTERED_E = ("primary", "wo", "primary_e02")   # readings fixed in advance; enriched ones are exploratory
BOOT_N, BOOT_SEED = 10_000, 20261009


def _rows(name: str) -> list[dict[str, str]]:
    with open(config.TABLES / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def table_e() -> list[dict[str, str]]:
    return _rows("E_controls.csv")


@lru_cache(maxsize=1)
def census() -> pd.DataFrame:
    return pd.DataFrame(_rows("R_reference_set.csv")).astype(
        {"points": int, "linked_points": int, "c_ans_declared": float, "c_ans_linked": float})


def _ratio(c: pd.DataFrame) -> float:
    return c.c_ans_linked.sum() / c.c_ans_declared.sum()


@lru_cache(maxsize=1)
def discount() -> dict[str, float]:
    c = census()
    pos = c[c.c_ans_declared > 0]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(pos), size=(BOOT_N, len(pos)))
    boot = pos.c_ans_linked.to_numpy()[idx].sum(1) / pos.c_ans_declared.to_numpy()[idx].sum(1)
    linked_series = pos[pos.linked_points > 0]
    return {
        "R1": _ratio(pos), "R2_low": float(np.percentile(boot, 2.5)),
        "R2_high": float(np.percentile(boot, 97.5)), "R3": _ratio(linked_series),
        "R4": float((pos.c_ans_linked / pos.c_ans_declared).median()),
        "n_pos": len(pos), "n_linked_series": len(linked_series),
        "n_declare_more": int((c.c_ans_declared > c.c_ans_linked + 1e-9).sum()),
    }


def r_levels(name: str = "R1") -> dict[str, float]:
    """r per level. R1 carries its bootstrap interval; R3 is a point alternative."""
    d = discount()
    if name == "R1":
        return {"low": d["R2_low"], "central": d["R1"], "high": d["R2_high"]}
    return {lv: d[name] for lv in floors.LEVELS}


# Census-supported bounds. Each keeps only the rows of its base reading whose every declared class
# appears in at least CENSUS_MIN of the readiness framework's 45 real automated-building graphs.
CENSUS_READINGS = {"census_primary": "primary", "census_enriched": "enriched"}
CENSUS_MIN = 0.10


@lru_cache(maxsize=1)
def census_share() -> dict[str, float]:
    c = census()
    dec = c.streams_declared.map(lambda s: set() if s == "-" else set(s.split("+")))
    classes = set().union(*dec)
    return {k: float(dec.map(lambda s, k=k: k in s).mean()) for k in classes}


def census_supported(row: dict) -> bool:
    share = census_share()
    return all(share.get(k, 0.0) >= CENSUS_MIN for k in row["streams"].split("+"))


def declared_streams(df: pd.DataFrame, reading: str) -> pd.Series:
    base = CENSUS_READINGS.get(reading, reading)
    tags = E_READINGS[base]
    have = {i: set() for i in df.index}
    for row in table_e():
        if row["reading"] not in tags or row["streams"] == "-" or row["id"] in E_EXCLUDE.get(base, ()):
            continue
        if reading in CENSUS_READINGS and not census_supported(row):
            continue
        hit = df.eval(row["condition"]).fillna(False).astype(bool)
        for i in df.index[hit]:
            have[i] |= set(row["streams"].split("+"))
    return pd.Series({i: frozenset(s) for i, s in have.items()})


def records(df: pd.DataFrame | None = None, floor_reading: str = PRIMARY["floor"],
            controls: str = PRIMARY["controls"], r: str = PRIMARY["r"]) -> pd.DataFrame:
    """floor / central / declared (C(I) and per-tier coverage) per record and level."""
    df = cbecs.load() if df is None else df
    rungs = floors.vintage.assign(df)
    rr = r_levels(r)
    dec = declared_streams(df, controls)
    out = {}
    for lv in floors.LEVELS:
        fl = floors.streams(df, rungs, floor_reading, lv)
        ds = pd.Series({i: fl[i] | dec[i] for i in df.index})
        measures = {"C": floors._score, **{t: (lambda s, t=t: tiers.coverage(s, t)) for t in tiers.TIERS}}
        for m, fn in measures.items():
            f_v, d_v = fl.map(fn), ds.map(fn)
            out[f"floor_{m}_{lv}"] = f_v
            out[f"declared_{m}_{lv}"] = d_v
            out[f"central_{m}_{lv}"] = f_v + (d_v - f_v) * rr[lv]
        out[f"floor_streams_{lv}"] = fl.map(lambda s: "+".join(sorted(s)) or "-")
        out[f"declared_streams_{lv}"] = ds.map(lambda s: "+".join(sorted(s)) or "-")
        out[f"cause_separating_{lv}"] = ds.map(tiers.cause_separating)
    return pd.DataFrame(out, index=df.index)


def configurations() -> dict[str, dict[str, str]]:
    """The primary configuration and each one-at-a-time alternative (the reference-set sensitivities)."""
    alts = {"primary": dict(PRIMARY)}
    for fr in floors.READINGS:
        if fr != PRIMARY["floor"]:
            alts[f"floor={fr}"] = {**PRIMARY, "floor": fr}
    for er in PRE_REGISTERED_E:
        if er != PRIMARY["controls"]:
            alts[f"controls={er}"] = {**PRIMARY, "controls": er}
    alts["r=R3"] = {**PRIMARY, "r": "R3"}
    return alts


BINS = [-0.001, 0.0005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
BIN_LABELS = ["0", "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]",
              "(0.6,0.7]", "(0.7,0.8]", "(0.8,1]"]
