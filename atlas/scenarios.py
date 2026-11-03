"""Scenario dimensions and code-timing analyses beside the primary result (c = d = 1, unchanged by everything here).

  design      description d applies only to recorded streams outside the BAS (Cfg.desc_scope = "nonbas", the default):
              BAS points already carry the description loss through their reference-set linkage. Grid over (0, 1] for
              compliance c and description d, tipping values, and Sobol indices on range sets (a) c, d in [0.5, 1] and
              (b) d in [0.05, 1]
  variant     one description draw per building (Cfg.desc_unit = "building"), same outputs, side by side
  timing      adoption lag of 1 and 2 years, and no monitoring credit outside 90.1 states (the sensitivity-table rows
              S15-S17 come from atlas.sensitivities): floor-area shares by imputed construction edition, the 2013+ share
              and cohort, and H3 (construction period minus building type) with full Rubin's rules
  S10 ref.    the same enriched reading as S10 without the credit (S10 itself is in atlas.sensitivities)
  BAS desc.   description d applied to every linked stream, BAS routes included (Cfg.desc_scope = "all", the earlier
              design), c = 1: grid over d in (0, 1] and tipping values (S23, d = 0.5, is in atlas.sensitivities)
  steps       floor-area shares at linked level exactly 0, 8/76, 12/76 and 20/76, and the four together

    python -m atlas.scenarios [M]   -> results/scenarios/*.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from atlas import canon, config, draws, logit_ci, mi, estimates, editions
from atlas.mi import Cfg

OUT = config.RESULTS / "scenarios"
WORKERS = config.WORKERS
VALS = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2))
UNITS = ("stream", "building")
LEVELS = {"0": 0, "8/76": 8, "12/76": 12, "20/76": 20}
TIMING = {"primary": {}, "lag 1 year": {"lag": 1}, "lag 2 years": {"lag": 2},
          "monitoring credit only under 90.1": {"monitoring_901_only": True}}


def _w4() -> float:
    R = pd.read_csv(estimates.OUT / "reads_decomposition.csv")
    return float(R[(R.scenario == "primary") & (R.group == "h4")].W.iloc[0])


# ── tasks 1-2 ────────────────────────────────────────────────────────────────────────────────────
def grids(M: int) -> None:
    w4 = _w4()
    for unit in UNITS:
        cfgs = sorted({Cfg(c=float(v), desc_unit=unit) for v in VALS} | {Cfg(d=float(v), desc_unit=unit) for v in VALS}
                      | {Cfg(c=c, d=d, desc_unit=unit) for c in (0.5, 0.75, 1.0) for d in (0.5, 0.75, 1.0)}, key=lambda k: (k.c, k.d))
        G = logit_ci.apply_frame(estimates.add_size_diffs(estimates.grid(M, cfgs, w4)), "scenario_grid.csv")
        G.to_csv(OUT / f"scenario_grid_{unit}.csv", index=False, float_format="%.6g")
        tip = estimates.tipping(G, "c", {"d": 1.0}) + estimates.tipping(G, "d", {"c": 1.0})
        pd.DataFrame(tip).to_csv(OUT / f"scenario_tipping_{unit}.csv", index=False)
        for tag, ranges in (("a", {"c": (0.5, 1.0), "d": (0.5, 1.0), "delta": (-3.0, 3.0)}),
                            ("b", {"c": (0.5, 1.0), "d": (0.05, 1.0), "delta": (-3.0, 3.0)})):
            old = dict(estimates.SOBOL_IN)
            estimates.SOBOL_IN.clear()
            estimates.SOBOL_IN.update(ranges)
            try:
                S = estimates.sobol(config.SOBOL_N, kw={"desc_unit": unit})
            finally:
                estimates.SOBOL_IN.clear()
                estimates.SOBOL_IN.update(old)
            S["max CI half-width"] = float(np.max(np.r_[(S["S1 hi"] - S["S1 lo"]) / 2, (S["ST hi"] - S["ST lo"]) / 2]))
            S["description unit"] = unit
            S.to_csv(OUT / f"sobol_{tag}_{unit}.csv", index=False, float_format="%.6g")


# ── tasks 3-5 and 7: one worker job per imputation ────────────────────────────────────────────────
def _job(m: int) -> tuple[int, list]:
    import warnings
    warnings.simplefilter("ignore")
    W = mi.weights()["area"]
    g_of = editions.group_of_rung()
    top = len(editions.GROUPS) - 1                      # "90.1-2013 or later"
    out = []
    for name, tm in TIMING.items():
        draws.TIMING.update(lag=tm.get("lag", 0), monitoring_901_only=tm.get("monitoring_901_only", False))
        try:
            imp = mi.make_imp("primary", m)
        finally:
            draws.TIMING.update(lag=0, monitoring_901_only=False)
        ev = mi.evaluate(imp, Cfg("primary"))
        grp = g_of[imp.new_rung]
        cols = {f"share under {g}": (grp == gi) * 1.0 for gi, g in enumerate(editions.GROUPS)}
        cols |= {k: v for k, v in mi.columns(ev, full=False).items()
                 if k in ("mean linked level", "status tier (linked)", "below the status tier (linked)", "linked level zero")}
        Q, U, names = editions._jack(cols, W, None)
        out.append((("timing", name, "stock"), names, Q, U))
        c13 = {k: v for k, v in mi.columns(ev, full=False).items() if k in ("mean linked level", "status tier (linked)", "linked level zero")}
        Q, U, names = editions._jack(c13, W, grp == top)
        out.append((("timing", name, "cohort 90.1-2013 or later"), names, Q, U))
        q4, u4 = mi.h4(imp, ev, jack=True)
        out.append((("timing", name, "H3"), ["construction period minus building type"], [q4], [u4]))
        if name == "primary":                        # the distribution's steps
            lv = np.rint(ev["sC"][:, 0] * 76).astype(int)
            steps = {f"linked level exactly {k}": (lv == v) * 1.0 for k, v in LEVELS.items()}
            steps["linked level at 0, 8/76, 12/76 or 20/76"] = np.isin(lv, list(LEVELS.values())) * 1.0
            Q, U, names = mi.jackknife(steps, W)
            out.append((("steps", "primary", "stock"), names, Q, U))
    imp = mi.make_imp("primary", m, extra_controls=("enriched_flagged",))      # S10 reference: S10's rows, no credit
    ev = mi.evaluate(imp, Cfg("primary", controls="enriched_flagged"))
    cols = {k: v for k, v in mi.columns(ev, full=False).items()
            if k in ("mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)",
                     "below the status tier (linked)", "linked level zero")}
    Q, U, names = mi.jackknife(cols, W)
    out.append((("s10 reference", "enriched rows with E20, not credited", "stock"), names, Q, U))
    return m, out


def run_jobs(M: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    acc: dict = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for m, out in ex.map(_job, range(M), chunksize=1):
            for g, names, Q, U in out:
                a = acc.setdefault(g, {"names": list(names), "Q": [], "U": []})
                a["Q"].append(np.asarray(Q, float))
                a["U"].append(np.asarray(U, float))
    frames = []
    for g, a in acc.items():
        T = mi.rubin(np.vstack(a["Q"]), np.vstack(a["U"]))
        T.insert(0, "statistic", a["names"])
        for i, col in enumerate(("part", "variant", "population")):
            T.insert(i, col, g[i])
        frames.append(T)
    return logit_ci.apply_frame(pd.concat(frames, ignore_index=True), "scenario_jobs.csv")


def levels_required() -> pd.DataFrame:
    """The three non-zero levels of the distribution's steps and the streams each requires (from the canon)."""
    rows = []
    for k, s in (("8/76", {"Whole"}), ("12/76", {"T", "Sp"}), ("20/76", {"T", "Sp", "Whole"})):
        c = canon.ceiling(s)
        cells = [f"{d} {i} L{canon.CANONS['v2'].achieved_level(d, i, s)}" for (d, i) in canon.CANONS["v2"].rungs
                 if canon.CANONS["v2"].achieved_level(d, i, s) > 0]
        rows.append({"level": k, "answer levels": c.achieved, "streams": "+".join(sorted(s)),
                     "means": {"8/76": "whole-building interval meter only", "12/76": "zone temperature and setpoint only",
                               "20/76": "zone temperature, setpoint and the whole-building interval meter"}[k],
                     "cells answered": "; ".join(cells)})
    return pd.DataFrame(rows)


def bas_description(M: int) -> None:
    """Description d applied to every linked stream, BAS routes included, over (0, 1] with c = 1, and where each conclusion
    flips along d."""
    cfgs = [Cfg(d=float(v), desc_scope="all") for v in VALS]
    G = logit_ci.apply_frame(estimates.add_size_diffs(estimates.grid(M, cfgs, _w4())), "scenario_grid.csv")
    G.to_csv(OUT / "scenario_grid_bas_description.csv", index=False, float_format="%.6g")
    pd.DataFrame(estimates.tipping(G, "d", {"c": 1.0})).to_csv(OUT / "scenario_tipping_bas_description.csv", index=False)


def main(M: int = mi.M_DEFAULT) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    grids(M)
    print(f"[{time.time() - t0:6.0f}s] grids and Sobol, both description units", flush=True)
    run_jobs(M).to_csv(OUT / "timing_steps_s10ref.csv", index=False, float_format="%.6g")
    levels_required().to_csv(OUT / "levels_required.csv", index=False)
    print(f"[{time.time() - t0:6.0f}s] code timing, S10 reference, steps", flush=True)
    bas_description(M)
    print(f"[{time.time() - t0:6.0f}s] grid with description on BAS streams too", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
