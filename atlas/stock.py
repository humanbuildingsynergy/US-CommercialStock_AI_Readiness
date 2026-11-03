"""Shared pieces of the stock-level analysis: the segment cuts, the Shapley decomposition of the floor-area-weighted R²
behind H3 (construction period against building type), and the stream bundle of the §8.4.3 monitoring clause.

Primary unit = floor area (FINALWT x SQFT). Sampling intervals are CBECS jackknife (``atlas.replicates``: 151
replicates, alpha = 1).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from atlas import canon, cbecs, distributions, floors
from atlas.replicates import REPLICATES

S9_FLAGS = ("ZYRCONC", "ZSQFT", "ZEMCS", "ZEMCSHT", "ZEMCSCL", "ZEMCSLT", "ZSMRTTHRM", "ZOCSN",
            "ZDCV", "ZVAV", "ZDAYHARV")   # CBECS imputation flags of the variables the model reads
CUTS = ("PBA", "SIZE", "SQFTC", "YRCONC", "PUBCLIM", "CENDIV", "OWNTYPE", "OWNOCC")


# ── H3: Shapley decomposition of the area-weighted R² ───────────────────────────────────────────
def _design(df: pd.DataFrame, factors: list[str]) -> np.ndarray:
    cols = [np.ones(len(df))]
    for f in factors:
        d = pd.get_dummies(df[f].astype(str), drop_first=True).to_numpy(dtype=float)
        cols.append(d)
    return np.column_stack(cols)


def _r2(X: np.ndarray, y: np.ndarray, a: np.ndarray) -> float:
    sw = np.sqrt(a)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = y - X @ beta
    ybar = (a * y).sum() / a.sum()
    return 1 - (a * res ** 2).sum() / (a * (y - ybar) ** 2).sum()


def shapley(df: pd.DataFrame, y: np.ndarray, factors: dict[str, str], W: np.ndarray, sqft: np.ndarray) -> pd.DataFrame:
    names = list(factors)
    K = len(names)
    designs = {S: _design(df, [factors[n] for n in S])
               for k in range(K + 1) for S in itertools.combinations(names, k)}
    out = []
    for col in range(W.shape[1]):
        a = W[:, col] * sqft
        r2 = {S: (0.0 if not S else _r2(X, y, a)) for S, X in designs.items()}
        phi = {}
        for n in names:
            tot = 0.0
            for k in range(K):
                for S in itertools.combinations([m for m in names if m != n], k):
                    Sn = tuple(m for m in names if m in S or m == n)
                    wgt = math.factorial(k) * math.factorial(K - k - 1) / math.factorial(K)
                    tot += wgt * (r2[Sn] - r2[tuple(m for m in names if m in S)])
            phi[n] = tot
        phi["R2 full"] = r2[tuple(names)]
        out.append(phi)
    return pd.DataFrame(out)


def h4(df: pd.DataFrame, y: np.ndarray, jack: bool = True) -> dict:
    """Shapley R² shares of *y* (one value per record) over construction period, type, size and (optionally) the
    reported-controls scope; 'vintage - type' is H3's quantity."""
    d = df.copy().assign(SIZE=floors.size_class(df.SQFT), SCOPE=distributions.emcs_scope(df))
    W = d[["FINALWT", *REPLICATES]].to_numpy(dtype=float) if jack else d[["FINALWT"]].to_numpy(dtype=float)
    sq = d.SQFT.to_numpy(dtype=float)
    res = {}
    for label, facs in (("with controls", {"vintage": "YRCONC", "type": "PBA", "size": "SIZE", "controls": "SCOPE"}),
                        ("without controls", {"vintage": "YRCONC", "type": "PBA", "size": "SIZE"})):
        S = shapley(d, y, facs, W, sq)
        S["vintage - type"] = S["vintage"] - S["type"]
        est = S.iloc[0]
        se = np.sqrt(((S.iloc[1:] - est) ** 2).sum()) if jack else est * np.nan
        res[label] = pd.DataFrame({"estimate": est, "se": se})
    return res


# ── H5: single-class lift ───────────────────────────────────────────────────────────────────────
H5_CLASSES = [c for c in canon.STREAMS if c not in ("Sub", "Sub2")] + ["Sub", "§8.4.3 bundle"]
BUNDLE = frozenset({"Lgt", "Pwr", "Sub", "Sub2"})


# ── cuts ────────────────────────────────────────────────────────────────────────────────────────
def cut_labels(df: pd.DataFrame, var: str) -> tuple[pd.Series, list[str]]:
    """Segment label per record, and the segments in code order."""
    if var == "SIZE":
        return floors.size_class(df.SQFT), list(floors.SIZE_CLASSES)
    codes = sorted(df[var].dropna().unique())
    order = [cbecs.label(var, c) for c in codes] + (["not applicable"] if df[var].isna().any() else [])
    return df[var].map(lambda c: "not applicable" if pd.isna(c) else cbecs.label(var, c)), order
