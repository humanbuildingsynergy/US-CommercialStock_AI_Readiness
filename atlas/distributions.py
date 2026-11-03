"""Descriptive report: unweighted and weighted distributions of every atlas variable.

Weighted columns use FINALWT (buildings) and FINALWT x SQFT (floor area). These are descriptive
counts of CBECS variables, not readiness numbers, so computing them before the analysis choices
were fixed could not bias those choices.

    python -m atlas.distributions      -> distributions.{md,csv} (an earlier design's outputs, not shipped)
"""
from __future__ import annotations

import pandas as pd

from atlas import cbecs, config


REPORT = ("YRCONC", "RENOV", "RENHVC", "RENLGT", "RENELC", "PBA", "PBAPLUS", "SQFTC", "PUBCLIM",
          "CENDIV", "OWNTYPE", "OWNOCC", "EMCS", "EMCSHT", "EMCSCL", "EMCSLT", "EMCS_SCOPE",
          "SMRTTHRM", "SCHED", "OCSN", "DIM", "LTCTL", "DCV", "VAV", "DOAS", "DAYHARV")


def emcs_scope(df: pd.DataFrame) -> pd.Series:
    """What the BAS controls, as one category (H heating, C cooling, L lighting)."""
    def one(r):
        if pd.isna(r.EMCS):
            return "not asked (unconditioned)"
        if r.EMCS == 2:
            return "no BAS"
        s = "".join(k for k, v in (("H", r.EMCSHT), ("C", r.EMCSCL), ("L", r.EMCSLT)) if v == 1)
        return "BAS: " + (s or "none of H/C/L")
    return df.apply(one, axis=1)


def lighting_controls(df: pd.DataFrame) -> pd.Series:
    """Which of OCSN / SCHED / DIM are reported, as one category."""
    def one(r):
        if pd.isna(r.OCSN):
            return "not asked"
        s = "+".join(k for k in ("OCSN", "SCHED", "DIM") if r[k] == 1)
        return s or "none"
    return df.apply(one, axis=1)


def table(df: pd.DataFrame, var: str) -> pd.DataFrame:
    if var == "EMCS_SCOPE":
        key = emcs_scope(df)
    elif var == "LTCTL":
        key = lighting_controls(df)
    else:
        key = df[var].map(lambda c: "not asked" if pd.isna(c) else int(c))
    area = df.FINALWT * df.SQFT
    g = pd.DataFrame({"key": key, "n": 1, "w": df.FINALWT, "a": area}).groupby("key", sort=True)
    t = g[["n", "w", "a"]].sum()
    out = pd.DataFrame({
        "variable": var,
        "code": t.index,
        "label": [k if isinstance(k, str) else cbecs.label(var, k) for k in t.index],
        "n": t.n.astype(int),
        "pct_n": 100 * t.n / t.n.sum(),
        "bldgs_k": t.w / 1e3,
        "pct_bldgs": 100 * t.w / t.w.sum(),
        "area_msqft": t.a / 1e6,
        "pct_area": 100 * t.a / t.a.sum(),
    }).reset_index(drop=True)
    return out


def imputed_share(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z in cbecs.FLAGS:
        v = z[1:]
        rows.append({"variable": v, "reported": int((df[z] == 0).sum()),
                     "imputed": int((df[z] == 1).sum()), "inapplicable": int((df[z] == 9).sum())})
    return pd.DataFrame(rows)
