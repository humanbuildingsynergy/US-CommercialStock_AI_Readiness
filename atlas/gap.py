"""The code-required -> declared -> linked gap and related analyses, from the same seeded imputations as
``atlas.estimates`` (``atlas.mi.make_imp`` rebuilds imputation m exactly), combined with Rubin's rules.

  gap.csv                  code-required → declared → linked; the declared-to-linked drop split exactly (Shapley) over
                           the thermostat rule, the meter rule, the BAS and network route rule and the BAS linkage draw;
                           description d = 1 contributes nothing. Area, energy and count weights.
  geography.csv            by census division and climate (PUBCLIM): mean linked level, tier shares, floor-area-weighted
                           advanced-meter share (expected AMI probability of the imputed state) and BAS share
  zero_profile.csv         the floor area whose linked level is zero: its distribution over size, type, period and
                           division, and its BAS, smart-thermostat and AMI-recorded-meter shares, beside the whole stock
  editions_imputed.csv     floor-area share under each imputed construction edition, by division and by period
  decomposition_scenarios.csv  Shapley decomposition by stream group for the three scenarios

    python -m atlas.gap [M]
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd

from atlas import draws, floors, mi, stock as stk, vintage
from atlas.estimates import Acc

OUT = mi.OUT
SWITCHES = ("rec_thermo", "rec_meter", "rec_route", "link")
SW_NAMES = {"rec_thermo": "thermostat rule", "rec_meter": "meter rule", "rec_route": "BAS and network route rule",
            "link": "BAS linkage draw"}


def ratio_jk(num: dict[str, np.ndarray], den: np.ndarray, w: np.ndarray):
    """Ratios of weighted sums Σ w·num_k / Σ w·den at the full weight and each replicate."""
    names = list(num)
    X = np.column_stack([num[k] for k in names])
    est = (w.T @ X) / (w.T @ den)[:, None]
    Q = est[0]
    return Q, ((est[1:] - Q) ** 2).sum(0), names


def gap_components(imp) -> dict[str, np.ndarray]:
    vals = {}
    for S in itertools.product((False, True), repeat=4):
        ev = mi.evaluate(imp, mi.Cfg("primary", **dict(zip(SWITCHES, S))))
        vals[S] = ev["sC"][:, 0]
        if all(S):
            full = ev
        if not any(S):
            assert np.array_equal(ev["C"], ev["D"])
    comp = {}
    for i, sw in enumerate(SWITCHES):
        tot = 0.0
        for S in itertools.product((False, True), repeat=4):
            if S[i]:
                continue
            k = sum(S)
            Sn = tuple(True if j == i else S[j] for j in range(4))
            tot = tot + math.factorial(k) * math.factorial(4 - k - 1) / math.factorial(4) * (vals[Sn] - vals[S])
        comp[f"drop: {SW_NAMES[sw]}"] = tot
    comp["code-required level"] = full["sF"][:, 0]
    comp["declared level"] = full["sD"][:, 0]
    comp["linked level"] = full["sC"][:, 0]
    comp["declared minus code-required"] = full["sD"][:, 0] - full["sF"][:, 0]
    comp["linked below code-required"] = (full["sC"][:, 0] < full["sF"][:, 0] - 1e-12) * 1.0
    comp["linked level zero"] = (full["sC"][:, 0] <= 1e-12) * 1.0
    return comp, full


def main(M: int = mi.M_DEFAULT) -> None:
    t0 = time.time()
    st = draws.stock()
    df = st.df
    W = mi.weights()
    th = 0.5 - 1e-12
    size = floors.size_class(df.SQFT).to_numpy()
    segs = {"SIZE": size, "PBA": df.PBA.to_numpy(), "YRCONC": df.YRCONC.to_numpy(), "CENDIV": df.CENDIV.to_numpy(),
            "PUBCLIM": df.PUBCLIM.to_numpy()}
    lab = {v: stk.cut_labels(df, v) for v in ("SIZE", "PBA", "YRCONC", "CENDIV", "PUBCLIM")}
    ladder = vintage.ladder()
    acc = Acc()
    for scen in mi.SCEN:
        for m in range(M):
            imp = mi.make_imp(scen, m)
            if scen == "primary":
                comp, ev = gap_components(imp)
                for wname in ("area", "energy", "count"):
                    Q, U, names = mi.jackknife(comp, W[wname])
                    acc.add(("gap", wname), names, Q, U)
                zero = (ev["sC"][:, 0] <= 1e-12) * 1.0
                cols = {"mean linked level": ev["sC"][:, 0], "status tier (linked)": (ev["sC"][:, 1] >= th) * 1.0,
                        "summary tier (linked)": (ev["sC"][:, 2] >= th) * 1.0, "diagnosis tier (linked)": (ev["sC"][:, 3] >= th) * 1.0,
                        "advanced-meter share (expected)": ev["ami_p"], "BAS share": ev["bas"] * 1.0, "linked level zero": zero}
                for var in ("CENDIV", "PUBCLIM"):
                    labels, order = lab[var]
                    for g in order:
                        mask = (labels == g).to_numpy()
                        Q, U, names = mi.jackknife(cols, W["area"], mask)
                        acc.add(("geo", var, g), names, Q, U)
                # zero profile: distribution of zero floor area over segments, and its BAS / smart / AMI shares
                for var in ("SIZE", "PBA", "YRCONC", "CENDIV"):
                    labels, order = lab[var]
                    num = {g: zero * (labels == g).to_numpy() for g in order}
                    Q, U, names = ratio_jk(num, zero, W["area"])
                    acc.add(("zero dist", var), names, Q, U)
                    Q, U, names = mi.jackknife({g: (labels == g).to_numpy() * 1.0 for g in order}, W["area"])
                    acc.add(("stock dist", var), names, Q, U)
                attrs = {"BAS": ev["bas"] * 1.0, "smart thermostat": ev["smart"] * 1.0, "AMI-recorded meter": ev["ami_hit"] * 1.0}
                Q, U, names = ratio_jk({k: v * zero for k, v in attrs.items()}, zero, W["area"])
                acc.add(("zero attrs",), names, Q, U)
                Q, U, names = mi.jackknife(attrs, W["area"])
                acc.add(("stock attrs",), names, Q, U)
                # imputed editions
                ind = {e: (ev["new_rung"] == k) * 1.0 for k, e in enumerate(ladder)}
                for var in ("CENDIV", "YRCONC"):
                    labels, order = lab[var]
                    for g in order:
                        Q, U, names = mi.jackknife(ind, W["area"], (labels == g).to_numpy())
                        acc.add(("editions", var, g), names, Q, U)
                Q, U, names = mi.jackknife(ind, W["area"])
                acc.add(("editions", "national", "all"), names, Q, U)
            else:
                ev = mi.evaluate(imp, mi.Cfg(scen))
            phi = {f"{lvl} {gname}": draws.phi(ev[key])[:, j] for lvl, key in (("code-required", "F"), ("declared", "D"), ("linked", "C"))
                   for j, gname in enumerate(draws.GROUP_NAMES)}
            Q, U, names = mi.jackknife(phi, W["area"])
            acc.add(("decomp", scen), names, Q, U)
        print(f"[{time.time() - t0:6.0f}s] gap {scen}", flush=True)

    def collect(pred, cols):
        out = []
        for g in acc.Q:
            if pred(g):
                T = acc.table(g)
                for i, c in enumerate(cols):
                    T.insert(i, c, g[i + 1] if i + 1 < len(g) else "")
                out.append(T)
        return pd.concat(out, ignore_index=True)
    collect(lambda g: g[0] == "gap", ["weight"]).to_csv(OUT / "gap.csv", index=False, float_format="%.6g")
    collect(lambda g: g[0] == "geo", ["cut", "segment"]).to_csv(OUT / "geography.csv", index=False, float_format="%.6g")
    zp = []
    for kind in ("zero dist", "stock dist"):
        T = collect(lambda g, k=kind: g[0] == k, ["cut"])
        T.insert(0, "population", "zero linked level" if kind.startswith("zero") else "whole stock")
        zp.append(T)
    for kind in ("zero attrs", "stock attrs"):
        T = acc.table((kind,))
        T.insert(0, "cut", "attribute")
        T.insert(0, "population", "zero linked level" if kind.startswith("zero") else "whole stock")
        zp.append(T)
    pd.concat(zp, ignore_index=True).to_csv(OUT / "zero_profile.csv", index=False, float_format="%.6g")
    collect(lambda g: g[0] == "editions", ["cut", "segment"]).to_csv(OUT / "editions_imputed.csv", index=False, float_format="%.6g")
    collect(lambda g: g[0] == "decomp", ["scenario"]).to_csv(OUT / "decomposition_scenarios.csv", index=False, float_format="%.6g")
    from atlas import logit_ci
    for name in ("gap.csv", "geography.csv", "zero_profile.csv", "editions_imputed.csv"):
        logit_ci.apply_file(OUT / name)   # shares' intervals on the logit scale
    print(f"[{time.time() - t0:6.0f}s] gap tables written", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
