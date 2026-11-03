"""Probability tables for the structural draws (exploratory). Every probability the draws use is a row
here, with its source; nothing is invented.

  tables/R_linkage_graphs.csv   per census graph: declared and linked classes, C_declared, C_linked,
                                the ratio C_linked / C_declared and its linkage type (all / none /
                                partial), over the R1 population (the 37 graphs with C_declared > 0)
  tables/R_linkage_classes.csv  per stream class: declared count, linked count and linked-given-
                                declared in the 37 graphs, in the 6 partial graphs and in all 45; the
                                probability the structural draw uses and its prior
  tables/E_controls_presence.csv  Table E with a presence probability per row: the share of the
                                45 census graphs that declare every class the row declares
  tables/C_compliance.csv       the 90.1-2013-onward clause rows of the primary floor reading, with the
                                compliance sweep c (sweep values, no source)

The census is the readiness framework's (copied with citation in ``tables/R_reference_set.csv``).

    python -m atlas.linkage      -> the four tables above
"""
from __future__ import annotations

import csv
from functools import lru_cache

import pandas as pd

from atlas import canon, config, declared, floors

MIN_DECL = 5                      # classes with fewer census declarations get a Beta prior
PRIORS = {"beta11": (1.0, 1.0), "beta0505": (0.5, 0.5)}
C_SWEEP = (1.0, 0.75, 0.5)        # compliance sweep values; c = 1.0 is the reference
COMPLIANCE_FROM = "2013"          # clauses from 90.1-2013 onward are subject to the sweep


def _fs(s) -> frozenset:
    return frozenset() if (not isinstance(s, str) or s == "-") else frozenset(s.split("+"))


@lru_cache(maxsize=1)
def graphs() -> pd.DataFrame:
    c = declared.census().copy()
    c["declared"] = c.streams_declared.map(_fs)
    c["linked"] = c.streams_linked.map(_fs)
    c["in_R1"] = c.c_ans_declared > 0
    ratio = (c.c_ans_linked / c.c_ans_declared).where(c.in_R1)
    c["ratio"] = ratio
    c["type"] = "-"
    c.loc[c.in_R1 & (ratio >= 1 - 1e-9), "type"] = "all"
    c.loc[c.in_R1 & (ratio <= 1e-9), "type"] = "none"
    c.loc[c.in_R1 & (ratio > 1e-9) & (ratio < 1 - 1e-9), "type"] = "partial"
    return c


def type_probs() -> dict[str, float]:
    g = graphs()
    pos = g[g.in_R1]
    return {t: float((pos.type == t).mean()) for t in ("all", "none", "partial")}


def type_counts() -> dict[str, int]:
    g = graphs()
    pos = g[g.in_R1]
    return {t: int((pos.type == t).sum()) for t in ("all", "none", "partial")}


def _counts(sub: pd.DataFrame, k: str) -> tuple[int, int]:
    dec = int(sub.declared.map(lambda s: k in s).sum())
    lin = int(sub.apply(lambda r: k in r.declared and k in r.linked, axis=1).sum())
    return dec, lin


@lru_cache(maxsize=1)
def class_table() -> pd.DataFrame:
    g = graphs()
    pos, part = g[g.in_R1], g[g.type == "partial"]
    rows = []
    for k in canon.STREAMS:
        d37, l37 = _counts(pos, k)
        d6, l6 = _counts(part, k)
        d45, l45 = _counts(g, k)
        row = {"class": k, "declared_R1": d37, "linked_R1": l37,
               "lgd_R1": (l37 / d37) if d37 else float("nan"),
               "declared_partial": d6, "linked_partial": l6,
               "lgd_partial": (l6 / d6) if d6 else float("nan"),
               "declared_all45": d45, "linked_all45": l45}
        if d37 >= MIN_DECL:
            row.update(p_used=l37 / d37, prior="none (empirical, >= 5 declarations)")
        else:
            a, b = PRIORS["beta11"]
            row.update(p_used=(a + l37) / (a + b + d37),
                       prior=f"Beta(1,1) prior + counts ({d37} declarations < {MIN_DECL}); drawn per draw")
        row["source"] = "the readiness framework's reference set (tables/R_reference_set.csv), the 37 graphs with C_declared > 0"
        rows.append(row)
    return pd.DataFrame(rows)


def population(name: str = "R1") -> pd.DataFrame:
    """R1: the 37 graphs with C_declared > 0. R3: those of them with at least one linked series (the S6
    population). partial: the 6 partial graphs (class rates only)."""
    g = graphs()
    pos = g[g.in_R1]
    return {"R1": pos, "R3": pos[pos.linked_points > 0], "partial": pos[pos.type == "partial"]}[name]


def fit(sub: pd.DataFrame, prior: str = "beta11", class_sub: pd.DataFrame | None = None) -> dict:
    """Type shares and per-class Beta parameters (or fixed rates) from a set of graphs. Used for the point
    model (sub = a population) and for each census bootstrap resample."""
    a, b = PRIORS[prior]
    n = len(sub)
    types = {t: float((sub.type == t).sum() / n) for t in ("all", "none", "partial")}
    cs = sub if class_sub is None else class_sub
    rates = {}
    for k in canon.STREAMS:
        d = int(cs.declared.map(lambda s, k=k: k in s).sum())
        l = int(cs.apply(lambda r, k=k: k in r.declared and k in r.linked, axis=1).sum()) if d else 0
        rates[k] = (l / d) if d >= MIN_DECL else (a + l, b + d - l)
    return {"types": types, "rates": rates}


def type_probs_pop(name: str = "R1") -> dict[str, float]:
    return fit(population(name))["types"]


def class_beta(prior: str = "beta11", population_name: str = "R1") -> dict[str, tuple[float, float] | float]:
    """Per class: a fixed probability (>= MIN_DECL declarations) or Beta(a + linked, b + unlinked)."""
    return fit(population(population_name), prior)["rates"]


def mean_rate(v) -> float:
    """Posterior mean of a class rate (the exact expectation over its Beta draw)."""
    return v if isinstance(v, float) else v[0] / (v[0] + v[1])


@lru_cache(maxsize=1)
def presence_table() -> pd.DataFrame:
    g = graphs()
    n = len(g)
    share = declared.census_share()
    rows = []
    for row in declared.table_e():
        out = dict(row)
        cls = _fs(row["streams"])
        if not cls:
            out.update(presence_k="", presence_n="", presence_p="", presence_prior="row declares nothing",
                       census_supported="", presence_source="")
        else:
            k = int(g.declared.map(lambda s: cls <= s).sum())
            if k >= MIN_DECL:
                p, prior = k / n, "none (empirical, >= 5 graphs)"
            else:
                p, prior = (1 + k) / (2 + n), f"Beta(1,1) prior + counts ({k} graphs < {MIN_DECL}); drawn per draw"
            out.update(presence_k=k, presence_n=n, presence_p=round(p, 6), presence_prior=prior,
                       census_supported=all(share.get(c, 0.0) >= declared.CENSUS_MIN for c in cls),
                       presence_source="share of the reference set's 45 graphs declaring every class of the row")
        rows.append(out)
    return pd.DataFrame(rows)


def presence_beta(prior: str = "beta11") -> dict[str, tuple[float, float] | float | None]:
    a, b = PRIORS[prior]
    out = {}
    for r in presence_table().itertuples():
        if r.presence_k == "":
            out[r.id] = None
        else:
            k, n = int(r.presence_k), int(r.presence_n)
            out[r.id] = (k / n) if k >= MIN_DECL else (a + k, b + n - k)
    return out


def compliance_rows() -> list[dict]:
    L = floors.vintage.ladder()
    need = L.index(COMPLIANCE_FROM)
    out = []
    for r in floors.table_c():
        if r["reading"] in floors.READINGS["main"] and L.index(r["rung_from"]) >= need:
            for c in C_SWEEP:
                out.append({"clause_row": r["id"], "clause": r["clause"], "rung_from": r["rung_from"],
                            "streams": r["streams"], "c": c,
                            "role": "reference" if c == 1.0 else "sweep",
                            "source": "sweep value, no source (no published installed-control rate)"})
    return out


def build() -> None:
    g = graphs()
    cols = ["source", "building", "in_R1", "type", "c_ans_declared", "c_ans_linked", "ratio",
            "streams_declared", "streams_linked"]
    g[cols].to_csv(config.TABLES / "R_linkage_graphs.csv", index=False, float_format="%.6f")
    class_table().to_csv(config.TABLES / "R_linkage_classes.csv", index=False, float_format="%.6f")
    presence_table().to_csv(config.TABLES / "E_controls_presence.csv", index=False)
    with open(config.TABLES / "C_compliance.csv", "w", encoding="utf-8", newline="") as f:
        rows = compliance_rows()
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    build()
    print("types:", type_counts())
    print(class_table()[["class", "declared_R1", "linked_R1", "lgd_R1", "declared_partial", "linked_partial", "p_used"]].to_string())
    print(presence_table()[["id", "streams", "presence_k", "presence_p", "census_supported"]].to_string())
