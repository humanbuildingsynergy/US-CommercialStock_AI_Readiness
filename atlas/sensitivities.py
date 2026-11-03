"""The sensitivity table: each sensitivity under the full multiple imputation of the primary result (M = 200, Rubin's
rules). Each sensitivity re-uses the primary imputations' random streams, so it differs from the primary only in what
it varies.

  S1  states weighted equally within each division          S7  reference set reweighted (infeasible: no graph has a
  S2  edition table without the implicit-practice rows          public type, size and period)
  S3  occupancy sensing from 90.1-2004 and 2010             S8  monthly bills credited
  S5  reported-controls table as first written              S9  smart thermostats not credited
      (BAS lighting -> occupancy)                           S10 the six flagged controls credited
  S6  rare-stream prior Beta(0.5, 0.5)                      S11 records with an imputed control answer dropped (contrast)
                                                            S12 advanced meters at the latest final EIA-861 year (2025)
                                                            S13 DOE's dates for the 17 pairs disputed with PNNL-31437
                                                            S14 interval metering for every building >= 25,000 ft2
  S15/S16 adoption lag of 1 / 2 years; S17 monitoring credit only under 90.1; S18 daily digital meter access
  S19 each domain weighted equally; S20 each intent weighted equally (these two also report the mean code-required and
      declared levels)
  S21 the 9 reference-set graphs that link none treated as missing; S22 only those of them with no time-series
      reference at all (both also report the code-required and declared means)
  S23 description d = 0.5 applied to BAS-route streams too
  S24 occupancy sensing credited from a 2016-rung lighting alteration (clause row C06, held out of the primary result)
(S4, a reading of 90.1-2022 whose source is not public, is not part of this release.)

Every statistic, H4 included, carries full Rubin's-rules intervals within its own sensitivity; the (sensitivity,
imputation) evaluations run in a process pool, each with its own random stream, so the result does not depend on the
number of workers.

    python -m atlas.sensitivities [M]              -> all rows
    python -m atlas.sensitivities [M] S12          -> only the named rows, merged into the existing sensitivities.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from atlas import config, draws, link_none, mi
from atlas.mi import Cfg
from atlas.estimates import CORE

OUT = mi.OUT
SENS = [
    ("S1", "States weighted equally within each division", {"state_weighted": False}, {}),
    ("S2", "Code-edition table without the rows assumed for states with no formal code", {"_drop_implicit": True}, {}),
    ("S3", "Occupancy sensing from 90.1-2004 and 2010 (timer-switch clauses)", {"floor_reading": "early_occ"}, {}),
    ("S5", "Reported-controls table as first written, with BAS lighting control read as occupancy sensing",
     {"extra_controls": ("wo",)}, {"controls": "wo"}),
    ("S6", "Rare-stream prior Beta(0.5, 0.5)", {"prior": "beta0505"}, {}),
    ("S7", "Reference set reweighted to the surveyed buildings with a BAS by size and type", None, None),
    ("S8", "Monthly bills credited", {}, {"rec_meter": False}),
    ("S9", "Smart thermostats not credited", {}, {"smart_credit": False}),
    ("S10", "Six reported controls credited as recorded without a BAS: CO2 sensors for demand-controlled ventilation, economizers, district-energy meters, demand-responsive lighting, VRF systems and energy-target tracking (on the enriched rows, so all six are declared)", {"extra_controls": ("enriched_flagged",)}, {"controls": "enriched_flagged", "flagged": "credited"}),
    ("S11", "Records with an imputed control answer dropped (contrast)", {}, {}),
    ("S12", "Advanced meters at 2025 shares (Form EIA-861 2025, latest final year)", {}, {"ami_year": 2025}),
    ("S13", "DOE's adoption dates for the 17 state-edition pairs where DOE and PNNL-31437 disagree", {"_doe_dates": True}, {}),
    ("S14", "Interval metering credited for every building of at least 25,000 ft2", {}, {"big_meter": True}),
    ("S15", "Adoption lag of 1 year: a building's edition is the one in force a year before its construction (and renovation) year",
     {"_lag": 1}, {}),
    ("S16", "Adoption lag of 2 years", {"_lag": 2}, {}),
    ("S17", "No credit for the 90.1-2013 monitoring clause where the state's code in force was an IECC edition or its own code",
     {"_mon901": True}, {}),
    ("S18", "Whole-building meter recorded with the state's share of commercial customers with daily digital access (2018), "
     "instead of the advanced-meter share", {}, {"meter_measure": "daily"}),
    ("S19", "Each of the six domains weighted equally (mean over domains of levels reached ÷ levels available)", {},
     {"scoring": "domain_equal"}),
    ("S20", "Each of the six intents weighted equally (mean over intents of levels reached ÷ levels available)", {},
     {"scoring": "intent_equal"}),
    # reference-set graphs treated as missing; the linkage model refitted without them
    ("S21", "The 9 reference-set knowledge graphs that link none treated as missing (linkage model fitted on the other 28)",
     {"drop_graphs": link_none.none_graphs()}, {}),
    ("S22", "Only the knowledge graphs that link none and publish no time-series reference treated as missing "
     f"({len(link_none.no_reference_graphs())} of 9; linkage model fitted on the other "
     f"{37 - len(link_none.no_reference_graphs())})", {"drop_graphs": link_none.no_reference_graphs()}, {}),
    # description d = 0.5 applied to BAS-route streams too (Cfg.desc_scope = "all", the earlier design)
    ("S23", "Description d = 0.5 applied to every linked stream, BAS routes included (the reference set's description "
     "not credited to a surveyed BAS)", {}, {"d": 0.5, "desc_scope": "all"}),
    # C06 is not in the primary result (exception 2 lets lamp/ballast and one-for-one retrofits comply by scheduled
    # shutoff); credited here. Same imputations as the primary (C06 is held in them).
    ("S24", "Occupancy sensing credited for lighting altered under 90.1-2016 (C06)", {}, {"alt_lighting": True}),
]
STATS = ("mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)",
         "below the status tier (linked)", "diagnosis tier (declared)", "linked level zero")


NAMES = [*STATS, "H2b concentration ratio", "H4 vintage - type"]
# the weighting sensitivities also report the code-required and declared means (the scoring changes them too); so do
# S21/S22 (unchanged by the linkage model, reported as a check) and S24 (C06 changes them)
EXTRA = {s: ("mean code-required level", "mean declared level") for s in ("S19", "S20", "S21", "S22", "S24")}


def _names(sid: str) -> list[str]:
    return [*NAMES, *EXTRA.get(sid, ())]
WORKERS = config.WORKERS


def _job(args: tuple) -> tuple[str, int, list, list]:
    """One (sensitivity, imputation): every statistic's point and within-imputation variance, H4 included with its own
    jackknife (full Rubin's rules within each sensitivity, no borrowed variance). Runs in a worker process."""
    import warnings
    warnings.simplefilter("ignore")
    sid, m = args
    ikw, ckw = SPEC[sid]
    kw = dict(ikw)
    drop, doe = kw.pop("_drop_implicit", False), kw.pop("_doe_dates", False)
    lag, mon = kw.pop("_lag", 0), kw.pop("_mon901", False)
    if drop or doe:
        draws._configure_vintage(drop_implicit=drop, doe_dates=doe)
    draws.TIMING.update(lag=lag, monitoring_901_only=mon)
    try:
        imp = mi.make_imp("primary", m, **kw)
    finally:
        draws.TIMING.update(lag=0, monitoring_901_only=False)
        if drop or doe:
            draws._configure_vintage()
    ev = mi.evaluate(imp, Cfg("primary", **ckw))
    W = mi.weights()["area"]
    cols = mi.columns(ev, full=False)
    mask = _keep() if sid == "S11" else None
    q, u, _ = mi.jackknife({k: cols[k] for k in STATS}, W, mask)
    qx, ux, _ = mi.jackknife({k: cols[k] for k in EXTRA.get(sid, ("mean linked level",))}, W, mask)
    if mask is None:
        h2 = mi.h2b(ev)
        h4 = mi.h4(imp, ev, jack=True)
    else:  # H2b on the kept records only; H4 not computed for the contrast
        th = 0.5 - 1e-12
        s = (ev["sC"][:, 1] >= th) * 1.0
        a = (W.T @ (s * mask)) / (W.T @ mask.astype(float))
        Wc = mi.weights()["count"]
        b = (Wc.T @ (s * mask)) / (Wc.T @ mask.astype(float))
        v = a / np.maximum(b, 1e-12)
        h2 = (float(v[0]), float(((v[1:] - v[0]) ** 2).sum()))
        h4 = (np.nan, np.nan)
    ex = len(EXTRA.get(sid, ()))
    return sid, m, [*q, h2[0], h4[0], *qx[:ex]], [*u, h2[1], h4[1], *ux[:ex]]


def _keep() -> np.ndarray:
    st = draws.stock()
    return ~(st.df[[f"Z{v}" for v in mi.IMPUTED_VARS]] == 1).any(axis=1).to_numpy()


SPEC = {"primary": ({}, {}), **{sid: (ikw, ckw) for sid, _, ikw, ckw in SENS if ikw is not None}}
LABEL = {"primary": "none (primary result)", **{sid: label for sid, label, _, _ in SENS}}


def main(M: int = mi.M_DEFAULT, only: tuple[str, ...] = ()) -> None:
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [s for s in SPEC if not only or s in only]
    jobs = [(s, m) for s in ids for m in range(M)]
    Q = {s: [None] * M for s in ids}
    U = {s: [None] * M for s in ids}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (sid, m, q, u) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            Q[sid][m], U[sid][m] = q, u
            if k % 400 == 0:
                print(f"[{time.time() - t0:6.0f}s] {k} of {len(jobs)} imputation evaluations", flush=True)
    rows = []
    for sid in ["primary", *[s[0] for s in SENS]]:
        if only and sid not in only:
            continue
        if sid not in SPEC:
            rows.append(pd.DataFrame([{"id": sid, "varied": LABEL[sid], "statistic": "not run",
                                       "note": "not possible: none of the 45 reference-set buildings has a public building type, "
                                               "size and construction period"}]))
            continue
        q, u = np.array(Q[sid], dtype=float), np.array(U[sid], dtype=float)
        if sid == "S11":
            q, u = q[:, :-1], u[:, :-1]
        T = mi.rubin(q, u)
        T.insert(0, "statistic", _names(sid)[: T.shape[0]] if sid != "S11" else NAMES[: T.shape[0]])
        if sid == "S11":
            keep = _keep()
            T["note"] = f"{int(keep.sum())} of {len(keep)} records kept; weights not recalibrated; H4 not computed"
        T.insert(0, "varied", LABEL[sid])
        T.insert(0, "id", sid)
        rows.append(T)
    S = pd.concat(rows, ignore_index=True)
    from atlas import logit_ci
    S["H4 note"] = np.where(S.statistic == "H4 vintage - type", "full Rubin's rules: each imputation's own jackknife variance", "")
    S = logit_ci.apply_frame(S, "sensitivities.csv")     # the new rows only: re-applying to rounded old rows drifts digits
    if only:  # merge into the full table, replacing the named rows and keeping the S-order; old rows are kept byte for byte
        old = pd.read_csv(OUT / "sensitivities.csv")
        S = pd.concat([old[~old.id.isin(only)], S], ignore_index=True)
        order = ["primary", *[s[0] for s in SENS]]
        S = S.sort_values("id", key=lambda c: c.map(order.index), kind="stable").reset_index(drop=True)
    S.to_csv(OUT / "sensitivities.csv", index=False, float_format="%.6g")
    print(f"[{time.time() - t0:6.0f}s] wrote sensitivities.csv", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT, tuple(sys.argv[2:]))
