"""Scoring and scenario analyses under the full multiple imputation (M = 200, Rubin's rules).

  scoring      the HVAC-credit scoring (equipment questions accept the points a BAS reads;
               tables/C_canon_hvac_credit.csv), for the primary result, both scenarios and the H1-H5 reads
  cells        domain-level answerability: the floor-area share reaching at least level 1 of every domain x intent cell,
               under both scorings, at the linked and declared levels; the three tier shares at the code-required,
               declared and linked levels
  optimistic   the optimistic scenario: the enriched scenario with a reference-set shift of +3 (types and class rates),
               the BAS-to-equipment-power row (E19) credited, lighting occupancy sensors recorded without BAS lighting
               control and an interval meter for every building of at least 25,000 ft2; under both scorings
  size tipping size-differential tipping: the reference-set shift for buildings below 25,000 ft2 only, or at or above it
               only

Each imputation runs in a worker process on the same random streams as the primary imputation m (common random numbers).

    python -m atlas.scoring [M]   -> results/scoring/*.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from atlas import config, draws, mi, stock as stk, tiers
from atlas.mi import Cfg

OUT = config.RESULTS / "scoring"
WORKERS = config.WORKERS
SCORINGS = ("primary", "hvac")
SCEN = ("strict (low)", "primary", "enriched (high)")
DELTAS = tuple(np.round(np.arange(-3, 3.0001, 0.5), 2))
OPTIMISTIC = dict(controls="optimistic", da=3.0, db=3.0, occ_any=True, big_meter=True)


def _tier_cols(ev: dict) -> dict[str, np.ndarray]:
    th = tiers.THRESHOLD - 1e-12
    out = {}
    for lvl, key in (("code-required", "sF"), ("declared", "sD"), ("linked", "sC")):
        out[f"mean {lvl} level"] = ev[key][:, 0]
        for j, t in enumerate(tiers.TIERS, 1):
            out[f"{t} tier ({lvl})"] = (ev[key][:, j] >= th) * 1.0
    out["below the status tier (linked)"] = (ev["sC"][:, 1] < th) * 1.0
    out["linked level zero"] = (ev["sC"][:, 0] <= 1e-12) * 1.0
    return out


def _cell_cols(ev: dict, scoring: str) -> dict[str, np.ndarray]:
    out = {}
    for lvl, key in (("declared", "D"), ("linked", "C")):
        L = draws.cell_levels(ev[key], scoring)
        for k, (d, i) in enumerate(draws.CELLS):
            out[f"{d} x {i} ({lvl})"] = (L[:, k] >= 1) * 1.0
    return out


def _job(m: int) -> tuple[int, list]:
    import warnings
    warnings.simplefilter("ignore")
    W = mi.weights()
    st = draws.stock()
    size_masks = {s: (st.sq < 25000) if s == "< 25,000" else (st.sq >= 25000) for s in ("< 25,000", ">= 25,000")}
    out = []

    def add(group, cols, w="area", mask=None):
        Q, U, names = mi.jackknife(cols, W[w], mask)
        out.append((group, names, Q, U))

    def h2(ev):
        Q, U = mi.h2b(ev)
        return [Q], [U]

    imps = {s: mi.make_imp(s, m, extra_controls=("optimistic",) if s == "enriched (high)" else ()) for s in SCEN}
    for sc in SCORINGS:
        for s in SCEN:
            ev = mi.evaluate(imps[s], Cfg(s, scoring=sc))
            add(("tiers", sc, s), _tier_cols(ev))
            Q, U = h2(ev)
            out.append((("h2b", sc, s), ["H2b concentration ratio"], Q, U))
            if s == "primary":
                add(("cells", sc, s), _cell_cols(ev, sc))
                for w in ("energy", "count"):
                    add(("tiers", sc, s, w), _tier_cols(ev), w)
                q4, u4 = mi.h4(imps[s], ev, jack=True)
                out.append((("h4", sc, s), ["H4 vintage - type"], [q4], [u4]))
                base = _tier_cols(ev)
                for c in stk.H5_CLASSES:
                    a_ = stk.BUNDLE | {"Whole"} if c == "§8.4.3 bundle" else frozenset({c})
                    c2 = _tier_cols(mi.evaluate(imps[s], Cfg(s, add=a_, scoring=sc)))
                    add(("h5", sc, c), {f"{t} gain": c2[f"{t} tier (linked)"] - base[f"{t} tier (linked)"] for t in ("summary", "diagnosis")})
        ev = mi.evaluate(imps["enriched (high)"], Cfg("enriched (high)", scoring=sc, **OPTIMISTIC))
        add(("tiers", sc, "optimistic"), _tier_cols(ev))
        Q, U = h2(ev)
        out.append((("h2b", sc, "optimistic"), ["H2b concentration ratio"], Q, U))
    for scope in ("small", "large"):            # size-differential reference-set shift (primary scoring)
        for d in DELTAS:
            ev = mi.evaluate(imps["primary"], Cfg("primary", da=float(d), db=float(d), delta_scope=scope))
            cols = _tier_cols(ev)
            add(("size tip", scope, float(d)), {k: cols[k] for k in ("mean linked level", "status tier (linked)", "below the status tier (linked)")})
            for k, msk in size_masks.items():
                add(("size tip mean", scope, float(d), k), {"mean linked level": cols["mean linked level"]}, mask=msk)
            Q, U = h2(ev)
            out.append((("size tip h2b", scope, float(d)), ["H2b concentration ratio"], Q, U))
    return m, out


def run(M: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    acc: dict = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (m, out) in enumerate(ex.map(_job, range(M), chunksize=1), 1):
            for g, names, Q, U in out:
                a = acc.setdefault(g, {"names": list(names), "Q": [], "U": []})
                a["Q"].append(np.asarray(Q, float))
                a["U"].append(np.asarray(U, float))
            if k % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] imputation {k} of {M}", flush=True)
    return acc


def table(a: dict) -> pd.DataFrame:
    R = mi.rubin(np.vstack(a["Q"]), np.vstack(a["U"]))
    R.insert(0, "statistic", a["names"])
    return R


def main(M: int = mi.M_DEFAULT) -> None:
    from atlas import logit_ci
    t0 = time.time()
    acc = run(M)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"scoring.csv": [], "cells.csv": [], "h5_scoring.csv": [], "size_tipping.csv": []}
    for g, a in acc.items():
        T = table(a)
        kind = g[0]
        if kind in ("tiers", "h2b", "h4"):
            T.insert(0, "weight", g[3] if len(g) > 3 else "area")
            T.insert(0, "scenario", g[2])
            T.insert(0, "scoring", g[1])
            frames["scoring.csv"].append(T)
        elif kind == "cells":
            T.insert(0, "scoring", g[1])
            frames["cells.csv"].append(T)
        elif kind == "h5":
            T.insert(0, "item", g[2])
            T.insert(0, "scoring", g[1])
            frames["h5_scoring.csv"].append(T)
        else:
            T.insert(0, "segment", g[3] if len(g) > 3 else "all")
            T.insert(0, "delta", g[2])
            T.insert(0, "shifted group", "< 25,000 ft2" if g[1] == "small" else ">= 25,000 ft2")
            frames["size_tipping.csv"].append(T)
    for name, fs in frames.items():
        D = pd.concat(fs, ignore_index=True)
        logit_ci.apply_frame(D, name).to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"[{time.time() - t0:6.0f}s] wrote scoring.csv, cells.csv, h5_scoring.csv, size_tipping.csv", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
