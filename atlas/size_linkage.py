"""Does building size predict how a reference-set knowledge graph links its declared streams?

The reference set's all / none / partial split (graphs whose declared streams answer any question, C_declared > 0) is
fitted as a multinomial logit on log10 floor area, with 'all' as the base category. Floor area is brick:area on the
brick:Building node of the Mortar graphs, recorded once in tables/R_linkage_size.csv with each graph file's sha256
(``extract_areas`` rebuilds it from a folder of the Mortar graphs). Test: likelihood ratio against the intercept-only
model (2 df); slopes with 95 % percentile bootstrap intervals over graphs (2,000 resamples, seeded).

    python -m atlas.size_linkage                 -> results/scoring/linkage_size.csv
    python -m atlas.size_linkage <mortar dir>    -> also rebuilds tables/R_linkage_size.csv from the graph files
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from atlas import config, linkage

OUT = config.RESULTS / "scoring"
TABLE = config.TABLES / "R_linkage_size.csv"
SEED = 20261017
TYPES = ("all", "none", "partial")


def extract_areas(mortar: Path) -> pd.DataFrame:
    g = linkage.graphs()
    rows = []
    for r in g.itertuples():
        f = Path(mortar) / f"{r.building}.ttl"
        area, sha = np.nan, ""
        if r.source == "mortar" and f.exists():
            t = f.read_text(encoding="utf-8", errors="replace")
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            m = re.search(r'brick:area\s*\[[^\]]*?brick:value\s*"([\d.]+)', t, re.S)
            if m:
                area = float(m.group(1))
        rows.append({"source": r.source, "building": r.building, "type": r.type, "in_reference_set": bool(r.in_R1),
                     "floor_area_ft2": area, "file_sha256": sha})
    return pd.DataFrame(rows)


def areas() -> pd.DataFrame:
    """The recorded floor areas, with each graph's linkage type and reference-set membership from the census table."""
    A = pd.read_csv(TABLE)
    g = linkage.graphs()[["source", "building", "type", "in_R1"]].rename(columns={"in_R1": "in_reference_set"})
    A = A.drop(columns=["type", "in_reference_set"]).merge(g, on=["source", "building"], how="right")
    return A[["source", "building", "type", "in_reference_set", "floor_area_ft2", "file_sha256"]]


def _nll(beta: np.ndarray, x: np.ndarray, y: np.ndarray, slope: bool = True) -> float:
    a = beta[:2]
    b = beta[2:] if slope else np.zeros(2)
    eta = np.column_stack([np.zeros(len(x)), a[0] + b[0] * x, a[1] + b[1] * x])
    eta -= eta.max(1, keepdims=True)
    lp = eta - np.log(np.exp(eta).sum(1, keepdims=True))
    return -lp[np.arange(len(y)), y].sum()


def fit(x: np.ndarray, y: np.ndarray, slope: bool = True) -> tuple[np.ndarray, float]:
    k = 4 if slope else 2
    r = optimize.minimize(_nll, np.zeros(k), args=(x, y, slope), method="BFGS")
    return r.x, r.fun


def main(mortar: str | None = None) -> None:
    if mortar:
        extract_areas(Path(mortar)).to_csv(TABLE, index=False)
    A = areas()
    d = A[A.in_reference_set & A.floor_area_ft2.notna()]
    x = np.log10(d.floor_area_ft2.to_numpy())
    xc = x - x.mean()
    y = d.type.map({t: i for i, t in enumerate(TYPES)}).to_numpy()
    b1, l1 = fit(xc, y)
    _, l0 = fit(xc, y, slope=False)
    lr = 2 * (l0 - l1)
    p = stats.chi2.sf(lr, 2)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 3:
            continue
        boots.append(fit(xc[i], y[i])[0][2:])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    big = d.floor_area_ft2 >= 25000
    tab = pd.crosstab(np.where(big, ">= 25,000 ft2", "< 25,000 ft2"), d.type)
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame([
        {"quantity": "graphs in the reference set with floor area", "value": len(d)},
        {"quantity": "median floor area (ft2)", "value": float(np.median(d.floor_area_ft2))},
        {"quantity": "slope, none vs all (per tenfold floor area)", "value": b1[2], "lo95": lo[0], "hi95": hi[0]},
        {"quantity": "slope, partial vs all (per tenfold floor area)", "value": b1[3], "lo95": lo[1], "hi95": hi[1]},
        {"quantity": "likelihood-ratio statistic (2 df)", "value": lr},
        {"quantity": "p-value", "value": p},
        {"quantity": "bootstrap resamples used", "value": len(boots)},
        *[{"quantity": f"count {s}, {t}", "value": int(tab.loc[s, t]) if t in tab.columns else 0}
          for s in tab.index for t in TYPES],
    ])
    res.to_csv(OUT / "linkage_size.csv", index=False, float_format="%.6g")
    print(res.to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
