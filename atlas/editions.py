"""AI readiness by imputed code edition, from the primary imputations (M = 200, Rubin's rules).

In each imputation every building joins an edition group by its construction edition (the Table V edition in force in
its imputed state and construction year). Variant: by the latest edition among its systems (construction, or the
renovation edition of a system the survey reports renovated since 2000). Groups follow the Table V ladder:

  before any 90.1-equivalent code (pre-1989) | 90.1-1989 | 90.1-1999/2001 | 90.1-2004 and 2007 | 90.1-2010 |
  90.1-2013 or later (2013, 2016)

Group membership varies across imputations, so each imputation's domain estimate and CBECS jackknife variance are
computed and then combined (Rubin's rules); shares take their intervals on the logit scale (atlas.logit_ci). Primary
result, floor-area weight, under the question set's scoring and the HVAC-credit scoring. Each group also reports its
survey records and Kish effective sample size (floor-area weights), averaged over imputations; groups below 30 records
are flagged.

    python -m atlas.editions [M]   -> results/editions/edition_levels.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from atlas import config, mi, vintage
from atlas.mi import Cfg

OUT = config.RESULTS / "editions"
WORKERS = config.WORKERS
SCORINGS = ("primary", "hvac")
BASES = ("construction", "latest")
GROUPS = ("before any 90.1-equivalent code", "90.1-1989", "90.1-1999/2001", "90.1-2004 and 2007", "90.1-2010",
          "90.1-2013 or later")
_EDITION_GROUP = {"pre-1989": 0, "1989": 1, "1999/2001": 2, "2004": 3, "2007": 3, "2010": 4, "2013": 5, "2016": 5,
                  "2019": 5, "2022": 5}
STATS = ("mean code-required level", "mean declared level", "mean linked level", "status tier (linked)", "linked level zero")
MIN_RECORDS = 30


def group_of_rung() -> np.ndarray:
    """Ladder rung index -> edition group index."""
    return np.array([_EDITION_GROUP[e] for e in vintage.ladder()])


def _jack(cols: dict, w: np.ndarray, seg: np.ndarray | None):
    """mi.jackknife, robust to a replicate in which the group has no weight (that replicate adds no variance)."""
    Q, U, names = mi.jackknife(cols, w, seg)
    if np.isnan(U).any():
        X = np.column_stack([cols[k] for k in names])
        A = w if seg is None else w * seg[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = (A.T @ X) / A.sum(0)[:, None]
        U = np.nansum((est[1:] - est[0]) ** 2, axis=0)
    return Q, U, names


def _job(m: int) -> tuple[int, list, dict]:
    import warnings
    warnings.simplefilter("ignore")
    W = mi.weights()["area"]
    g_of = group_of_rung()
    imp = mi.make_imp("primary", m)
    out, counts = [], {}
    member = {b: g_of[imp.new_rung if b == "construction" else imp.latest_rung] for b in BASES}
    evs = {sc: mi.evaluate(imp, Cfg("primary", scoring=sc)) for sc in SCORINGS}
    for b in BASES:
        for gi, g in enumerate(GROUPS):
            mask = member[b] == gi
            n_rec = int(mask.sum())
            w0 = W[mask, 0]
            counts[(b, g)] = (n_rec, float(w0.sum() ** 2 / (w0 ** 2).sum()) if n_rec else 0.0)
            if n_rec == 0:
                continue
            Q, U, names = _jack({"share of floor area": mask * 1.0}, W, None)
            out.append(((b, "primary", g, "share"), names, Q, U))   # the share does not depend on the scoring
            for sc, ev in evs.items():
                cols = {k: v for k, v in mi.columns(ev, full=False).items() if k in STATS}
                Q, U, names = _jack(cols, W, mask)
                out.append(((b, sc, g, "levels"), names, Q, U))
    return m, out, counts


def run(M: int):
    from concurrent.futures import ProcessPoolExecutor
    from atlas.estimates import Acc
    t0 = time.time()
    acc, rec = Acc(), {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (m, out, counts) in enumerate(ex.map(_job, range(M), chunksize=1), 1):
            for key, names, Q, U in out:
                acc.add(key, names, Q, U)
            for key, v in counts.items():
                rec.setdefault(key, []).append(v)
            if k % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] imputation {k} of {M}", flush=True)
    return acc, rec


def frame(acc, rec) -> pd.DataFrame:
    from atlas import logit_ci
    rows = []
    for key in acc.Q:
        b, sc, g, _ = key
        T = acc.table(key)
        n = np.array(rec[(b, g)], float)
        T.insert(0, "group", g)
        T.insert(0, "scoring", sc)
        T.insert(0, "basis", b)
        T["records (mean over imputations)"] = n[:, 0].mean()
        T["Kish effective n (mean over imputations)"] = n[:, 1].mean()
        T["imputations with the group"] = int((n[:, 0] > 0).sum())
        T["fewer than 30 records"] = bool(n[:, 0].mean() < MIN_RECORDS)
        rows.append(T)
    D = pd.concat(rows, ignore_index=True)
    order = {g: i for i, g in enumerate(GROUPS)}
    D = D.sort_values(["basis", "scoring", "group"], key=lambda s: s.map(order) if s.name == "group" else s, kind="stable")
    return logit_ci.apply_frame(D, "edition_levels.csv")


def main(M: int = mi.M_DEFAULT) -> None:
    t0 = time.time()
    acc, rec = run(M)
    OUT.mkdir(parents=True, exist_ok=True)
    D = frame(acc, rec)
    D.to_csv(OUT / "edition_levels.csv", index=False, float_format="%.6g")
    print(f"[{time.time() - t0:6.0f}s] wrote results/editions/edition_levels.csv", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
