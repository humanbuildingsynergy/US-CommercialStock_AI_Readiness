"""Main estimates: multiple imputation (national, cut and contrast tables), scenario dimensions, the transfer
tipping point, model checks and the Sobol analysis. Outputs in results/estimates/.

    python -m atlas.estimates [M]
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from atlas import canon, config, draws, floors, linkage, mi, stock as stk, tiers
from atlas.mi import Cfg

OUT = mi.OUT
CORE = ("mean code-required level", "mean declared level", "mean linked level", "status tier (linked)",
        "summary tier (linked)", "diagnosis tier (linked)", "below the status tier (linked)", "diagnosis tier (declared)",
        "linked below code-required", "linked level zero", "at the minimal code-required profile")
CUT_COLS = ("mean code-required level", "mean declared level", "mean linked level", "status tier (linked)",
            "summary tier (linked)", "diagnosis tier (linked)")


@np.vectorize
def _noop(x):
    return x


class Acc:
    """Collects (Q, U) per imputation for a set of named statistics, then applies Rubin's rules."""

    def __init__(self):
        self.Q, self.U, self.names = {}, {}, {}

    def add(self, group, names, Q, U):
        self.names.setdefault(group, list(names))
        self.Q.setdefault(group, []).append(np.asarray(Q, float))
        self.U.setdefault(group, []).append(np.asarray(U, float))

    def table(self, group, M: int | None = None) -> pd.DataFrame:
        Q = np.vstack(self.Q[group][:M])
        U = np.vstack(self.U[group][:M])
        R = mi.rubin(Q, U)
        R.insert(0, "statistic", self.names[group])
        return R


def _cut_masks(df):
    out = {}
    for var in stk.CUTS:
        lab, order = stk.cut_labels(df, var)
        for g in order:
            out[(var, g)] = (lab == g).to_numpy()
    return out


def imputations(scenario: str, M: int, log: dict | None = None) -> list:
    return [mi.make_imp(scenario, m, log=log) for m in range(M)]


# ── multiple imputation ──────────────────────────────────────────────────────────────────────────
WORKERS = config.WORKERS


def _pool():
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=WORKERS)


class _Collect:
    """Stands in for Acc inside a worker: records every add() for the parent to replay in imputation order."""

    def __init__(self):
        self.out = []

    def add(self, group, names, Q, U):
        self.out.append((group, list(names), np.asarray(Q, float), np.asarray(U, float)))


def _mi_job(args: tuple) -> tuple[int, list, dict]:
    """One imputation of one scenario, in a worker process. The imputation's random stream depends only on
    (scenario, m), so the result does not depend on the number of workers."""
    import warnings
    warnings.simplefilter("ignore")
    scenario, m, primary = args
    log: dict = {}
    imp = mi.make_imp(scenario, m, log=log)
    acc = _Collect()
    _mi_body(acc, scenario, imp, primary)
    return m, acc.out, log


def run_mi(scenario: str, M: int, primary: bool = False) -> tuple[Acc, None, dict]:
    acc, log = Acc(), {}
    with _pool() as ex:
        res = sorted(ex.map(_mi_job, [(scenario, m, primary) for m in range(M)], chunksize=2), key=lambda r: r[0])
    for _, out, lg in res:
        for g, names, Q, U in out:
            acc.add(g, names, Q, U)
        for k, v in lg.items():
            log[k] = log.get(k, 0) + v
    return acc, None, log


def _mi_body(acc, scenario: str, imp, primary: bool) -> None:
    st = draws.stock()
    W = mi.weights()
    cuts = _cut_masks(st.df)
    imputed = (st.df[[f"Z{v}" for v in mi.IMPUTED_VARS]] == 1).any(axis=1).to_numpy()
    if True:
        ev = mi.evaluate(imp, Cfg(scenario))
        cols = mi.columns(ev, full=True)
        for wname in ("area", "energy", "count"):
            Q, U, names = mi.jackknife(cols, W[wname])
            acc.add(("national", wname), names, Q, U)
        core = {k: cols[k] for k in CUT_COLS}
        for (var, g), mask in cuts.items():
            for wname in (("area", "energy", "count") if var in ("SIZE", "YRCONC", "PBA") else ("area",)):
                Q, U, names = mi.jackknife(core, W[wname], mask)
                acc.add(("cut", wname, var, g), names, Q, U)
        Q, U = mi.h2b(ev)
        acc.add(("h2b",), ["status tier: floor-area share / building share"], [Q], [U])
        Q, U, names = mi.jackknife({k: cols[k] for k in CORE}, W["area"], ~imputed)
        acc.add(("imputed-answer contrast",), names, Q, U)
        if primary:
            q4, u4 = mi.h4(imp, ev, jack=True)
            acc.add(("h4",), ["vintage - type (with controls)"], [q4], [u4])
            phi = {f"{lvl} {gname}": draws.phi(ev[key])[:, j] for lvl, key in (("code-required", "F"), ("declared", "D"), ("linked", "C"))
                   for j, gname in enumerate(draws.GROUP_NAMES)}
            Q, U, names = mi.jackknife(phi, W["area"])
            acc.add(("decomposition",), names, Q, U)
            base = {"summary": cols["summary tier (linked)"], "diagnosis": cols["diagnosis tier (linked)"], "mean": cols["mean linked level"]}
            for c in stk.H5_CLASSES:
                add = stk.BUNDLE | {"Whole"} if c == "§8.4.3 bundle" else frozenset({c})
                e2 = mi.evaluate(imp, Cfg(scenario, add=add))
                c2 = mi.columns(e2, full=False)
                Q, U, names = mi.jackknife({"summary gain": c2["summary tier (linked)"] - base["summary"],
                                            "diagnosis gain": c2["diagnosis tier (linked)"] - base["diagnosis"]}, W["area"])
                acc.add(("h5", c), names, Q, U)
            from atlas import bounds
            for label, add, scope in (("bundle, every building", bounds.BUNDLE, "all"), ("bundle, >= 25,000 ft2 only", bounds.BUNDLE, "25k"),
                                      ("bundle without equipment power, every building", bounds.BUNDLE_NO_PWR, "all"),
                                      ("bundle without equipment power, >= 25,000 ft2 only", bounds.BUNDLE_NO_PWR, "25k")):
                e2 = mi.evaluate(imp, Cfg(scenario, add=add | {"Whole"}, add_scope=scope))
                c2 = mi.columns(e2, full=False)
                Q, U, names = mi.jackknife({"mean linked gain": c2["mean linked level"] - base["mean"],
                                            "summary gain": c2["summary tier (linked)"] - base["summary"],
                                            "diagnosis gain": c2["diagnosis tier (linked)"] - base["diagnosis"]}, W["area"])
                acc.add(("lever", label), names, Q, U)


# ── scenario and transfer grids re-using the primary imputations (common random numbers) ────────
def core_eval(imp, cfg: Cfg, size_masks: dict, w4: float | None = None) -> dict:
    """Core statistics (Q, U) for one imputation and grid point; H4 is a point estimate (U from the primary W̄)."""
    ev = mi.evaluate(imp, cfg)
    cols = mi.columns(ev, full=False)
    W = mi.weights()["area"]
    Q, U, names = mi.jackknife({k: cols[k] for k in CORE}, W)
    out = dict(zip(names, zip(Q, U)))
    out["H2b"] = mi.h2b(ev)
    for s, mask in size_masks.items():
        q, u, _ = mi.jackknife({"m": cols["mean linked level"]}, W, mask)
        out[f"mean linked, {s}"] = (q[0], u[0])
    out["H4"] = (mi.h4(imp, ev, jack=False)[0], w4 if w4 is not None else np.nan)
    return out


def _grid_job(args: tuple) -> tuple[int, list[dict]]:
    import warnings
    warnings.simplefilter("ignore")
    m, cfgs, w4 = args
    st = draws.stock()
    size_masks = {s: (floors.size_class(st.df.SQFT) == s).to_numpy() for s in floors.SIZE_CLASSES}
    imp = mi.make_imp("primary", m)
    return m, [core_eval(imp, cfg, size_masks, w4) for cfg in cfgs]


def grid(M: int, cfgs: list[Cfg], w4: float) -> pd.DataFrame:
    """Each configuration over the primary imputations 0..M-1 (common random numbers), evaluated in worker processes."""
    with _pool() as ex:
        res = sorted(ex.map(_grid_job, [(m, cfgs, w4) for m in range(M)], chunksize=2), key=lambda r: r[0])
    rows = []
    for k_, cfg in enumerate(cfgs):
        per = [r[1][k_] for r in res]
        keys = list(per[0])
        Q = np.array([[p[k][0] for k in keys] for p in per])
        U = np.array([[p[k][1] for k in keys] for p in per])
        R = mi.rubin(Q, U)
        R.insert(0, "statistic", keys)
        for k, v in cfg.__dict__.items():
            if k not in ("add", "add_scope"):
                R[k] = v
        rows.append(R)
    return pd.concat(rows, ignore_index=True)


CONCLUSIONS = {  # name: (statistic, direction, boundary) — the conclusion holds when the statistic is on that side
    "most floor area reaches no readiness tier (H1)": ("below the status tier (linked)", ">", 0.5),
    "readiness concentrated in large buildings (H2b)": ("H2b", ">", 1.0),
    "a minority of floor area is diagnosis-ready at the declared level (H3)": ("diagnosis tier (declared)", "<", 0.5),
    "building type explains more than construction period (H4)": ("H4", "<", 0.0),
    "small < mid-size buildings (size ranking)": ("size: small - mid", "<", 0.0),
    "mid-size < large buildings (size ranking)": ("size: mid - large", "<", 0.0),
}


def add_size_diffs(G: pd.DataFrame) -> pd.DataFrame:
    """Size-ranking contrasts from the per-size means (point and T; the ranks are compared conservatively)."""
    keys = [c for c in G.columns if c not in ("statistic", "estimate", "W", "B", "T", "se", "df", "fmi", "lo95", "hi95", "M")]
    rows = []
    for gk, g in G.groupby(keys, dropna=False):
        g = g.set_index("statistic")
        for name, a, b in (("size: small - mid", "<= 5,000", "5,001-24,999"), ("size: mid - large", "5,001-24,999", ">= 25,000")):
            ea, eb = g.loc[f"mean linked, {a}"], g.loc[f"mean linked, {b}"]
            T = ea["T"] + eb["T"]
            est = ea.estimate - eb.estimate
            rows.append({"statistic": name, "estimate": est, "T": T, "se": np.sqrt(T), "lo95": est - 1.96 * np.sqrt(T),
                         "hi95": est + 1.96 * np.sqrt(T), **dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))})
    return pd.concat([G, pd.DataFrame(rows)], ignore_index=True)


def tipping(G: pd.DataFrame, axis: str, fixed: dict) -> list[dict]:
    """For each conclusion: the first value along *axis* (moving away from the reference) at which the point estimate
    flips and at which the 95 % interval first includes the opposite conclusion."""
    out = []
    sel = G
    for k, v in fixed.items():
        sel = sel[np.isclose(sel[k], v)]
    for cname, (stat, dirn, bound) in CONCLUSIONS.items():
        s = sel[sel.statistic == stat].sort_values(axis)
        if s.empty:
            continue
        holds_pt = (s.estimate > bound) if dirn == ">" else (s.estimate < bound)
        holds_ci = (s.lo95 > bound) if dirn == ">" else (s.hi95 < bound)
        ref = 1.0 if axis in ("c", "d") else 0.0
        row = {"conclusion": cname, "axis": axis, **fixed}
        for side, order in (("down", s[s[axis] <= ref + 1e-9].sort_values(axis, ascending=False)),
                            ("up", s[s[axis] >= ref - 1e-9].sort_values(axis))):
            if order.empty:
                continue
            hp = ((order.estimate > bound) if dirn == ">" else (order.estimate < bound)).to_numpy()
            hc = ((order.lo95 > bound) if dirn == ">" else (order.hi95 < bound)).to_numpy()
            vals = order[axis].to_numpy()
            row[f"point flips ({side})"] = vals[np.argmax(~hp)] if (~hp).any() else "none in range"
            row[f"interval includes opposite ({side})"] = vals[np.argmax(~hc)] if (~hc).any() else "none in range"
        row["holds at reference"] = bool(holds_pt[np.isclose(s[axis], ref)].all())
        out.append(row)
    return out


# ── model checks ─────────────────────────────────────────────────────────────────────────────────
def tails(sim: np.ndarray, obs: float) -> dict:
    """Posterior predictive tail probabilities over the simulated reference sets (NaN draws left out): Pr(sim >= obs),
    Pr(sim <= obs) and the two-sided value 2 x min of the two, capped at 1."""
    a = sim[~np.isnan(sim)]
    up, lo = float((a >= obs - 1e-9).mean()), float((a <= obs + 1e-9).mean())
    return {"Pr(sim >= obs)": up, "Pr(sim <= obs)": lo, "two-sided p": min(1.0, 2 * min(up, lo))}


def model_checks(n_sim: int = 2000, detail: list | None = None) -> pd.DataFrame:
    """Simulate the census from the fitted imputation model, with parameter draws, and compare. ``detail``, when given,
    receives one (simulation, declared streams, streams drawn, linked share) per graph drawn as the partial type; the
    random draws are the same either way."""
    g = linkage.graphs()
    pos = g[g.in_R1].reset_index(drop=True)
    rng = mi._rng("model-check")
    dec, lin, ty = mi._census_arrays()
    sims = {"all": [], "none": [], "partial": [], "partial ratio min": [], "partial ratio max": []}
    cls_dec = {c: [] for c in canon.STREAMS}
    cls_lin = {c: [] for c in canon.STREAMS}
    decl_sets = list(pos.declared)
    for s in range(n_sim):
        types, q = mi.draw_params(rng)
        t = rng.choice(3, size=len(pos), p=types)
        ratios = []
        lin_counts = {c: 0 for c in canon.STREAMS}
        for i, ds in enumerate(decl_sets):
            if t[i] == 0:
                ls = set(ds)
            elif t[i] == 1:
                ls = set()
            else:
                # sorted: one draw per stream in a fixed order (a set's order changes with Python's hash seed)
                ls = {c for c in sorted(ds) if rng.random() < q[draws.CLS.index(c)]}
            cd = canon.ceiling(ds).score
            r_ = canon.ceiling(ls).score / cd if cd > 0 else np.nan
            ratios.append(r_)
            if detail is not None and t[i] == 2:
                detail.append((s, len(ds), len(ls), r_))
            for c in ls:
                lin_counts[c] += 1
        ratios = np.array(ratios)
        sims["all"].append(int((ratios >= 1 - 1e-9).sum()))
        sims["none"].append(int((ratios <= 1e-9).sum()))
        part = ratios[(ratios > 1e-9) & (ratios < 1 - 1e-9)]
        sims["partial"].append(len(part))
        sims["partial ratio min"].append(part.min() if len(part) else np.nan)
        sims["partial ratio max"].append(part.max() if len(part) else np.nan)
        for c in canon.STREAMS:
            cls_lin[c].append(lin_counts[c])
    obs = {"all": 22, "none": 9, "partial": 6}
    pr = pos[pos.type == "partial"].ratio
    obs["partial ratio min"], obs["partial ratio max"] = float(pr.min()), float(pr.max())
    rows = []
    for k, v in sims.items():
        a = np.array(v, float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        rows.append({"statistic": k, "observed": obs[k], "simulated 2.5%": lo, "simulated 97.5%": hi,
                     "inside": bool(lo - 1e-9 <= obs[k] <= hi + 1e-9), **tails(a, obs[k])})
    for c in canon.STREAMS:
        d = int(pos.declared.map(lambda s: c in s).sum())
        if d == 0:
            continue
        l_obs = int(pos.apply(lambda r: c in r.declared and c in r.linked, axis=1).sum())
        a = np.array(cls_lin[c], float)
        lo, hi = np.percentile(a, [2.5, 97.5])
        rows.append({"statistic": f"linked count, {c} (declared {d})", "observed": l_obs, "simulated 2.5%": lo,
                     "simulated 97.5%": hi, "inside": bool(lo <= l_obs <= hi), **tails(a, l_obs)})
    return pd.DataFrame(rows)



def model_check_partial(detail: list) -> dict[str, pd.DataFrame]:
    """Why the simulated number of partly linked graphs rarely exceeds the observed 6. A graph drawn as the partial type
    counts as partly linked only if its linked share of answer levels ends strictly between 0 and 1; by number of
    declared streams, how many end at 0, at 1 or strictly between, and how many have none or all of their streams drawn.
    ``detail`` is the list model_checks(detail=...) filled."""
    D = pd.DataFrame(detail, columns=["sim", "declared", "drawn", "share"])
    n_sim = int(D.sim.max()) + 1 if len(D) else 0
    D["ends at 0"] = D.share <= 1e-9
    D["ends at 1"] = D.share >= 1 - 1e-9
    D["strictly between"] = ~(D["ends at 0"] | D["ends at 1"])
    D["none drawn"] = D.drawn == 0
    D["all drawn"] = D.drawn == D.declared
    rows = []
    for k, g in [*D.groupby("declared"), ("all", D)]:
        rows.append({"declared streams": k, "graphs (all simulations)": len(g), "per simulated reference set": len(g) / n_sim,
                     **{f"share {c}": float(g[c].mean()) for c in ("ends at 0", "ends at 1", "strictly between",
                                                                  "none drawn", "all drawn")}})
    per = D.groupby("sim").agg(partial_class=("share", "size"), between=("strictly between", "sum")).reindex(range(n_sim), fill_value=0)
    C = per.between.value_counts().sort_index().rename_axis("partly linked graphs").reset_index(name="simulations")
    C["share of simulations"] = C.simulations / n_sim
    means = pd.DataFrame([{"simulations": n_sim, "mean partial-class graphs": per.partial_class.mean(),
                           "mean strictly between": per.between.mean()}])
    return {"model_check_partial.csv": pd.DataFrame(rows), "partial_count.csv": C, "partial_means.csv": means}

# ── Sobol over c, d and δ ────────────────────────────────────────────────────────────────────────
SOBOL_IN = {"c": (0.5, 1.0), "d": (0.5, 1.0), "delta": (-3.0, 3.0)}
SOBOL_OUT = ("mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)")


SOBOL_M = 4   # imputations averaged per model evaluation
_SOBOL_IMPS: list = []


def _sobol_job(args) -> np.ndarray:
    import warnings
    warnings.simplefilter("ignore")
    X, kw = args
    if not _SOBOL_IMPS:
        _SOBOL_IMPS.extend(mi.make_imp("primary", m) for m in range(SOBOL_M))
    return np.array([sobol_eval(_SOBOL_IMPS, x, kw) for x in X])


def _sobol_rows(X: np.ndarray, kw: dict | None = None) -> np.ndarray:
    with _pool() as ex:
        return np.vstack(list(ex.map(_sobol_job, [(x, kw or {}) for x in np.array_split(X, max(1, len(X) // 128))])))


def sobol_eval(imps: list, x: np.ndarray, kw: dict | None = None) -> np.ndarray:
    """kw: further Cfg fields held fixed over the design (e.g. desc_scope, desc_unit in atlas.scenarios)."""
    W = mi.weights()["area"][:, 0]
    out = np.zeros(len(SOBOL_OUT))
    for imp in imps:
        ev = mi.evaluate(imp, Cfg("primary", c=x[0], d=x[1], da=x[2], db=x[2], **(kw or {})))
        cols = mi.columns(ev, full=False)
        out += np.array([(W * cols[k]).sum() / W.sum() for k in SOBOL_OUT])
    return out / len(imps)


def sobol(N: int, n_boot: int = 500, kw: dict | None = None) -> pd.DataFrame:
    rng = mi._rng("sobol")
    k = len(SOBOL_IN)
    lo = np.array([v[0] for v in SOBOL_IN.values()])
    hi = np.array([v[1] for v in SOBOL_IN.values()])
    A = lo + (hi - lo) * rng.random((N, k))
    B = lo + (hi - lo) * rng.random((N, k))
    ABs = []
    for i in range(k):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        ABs.append(ABi)
    F = _sobol_rows(np.vstack([A, B, *ABs]), kw)
    fA, fB = F[:N], F[N:2 * N]
    fAB = [F[(2 + i) * N:(3 + i) * N] for i in range(k)]

    def indices(idx):
        V = np.var(np.vstack([fA[idx], fB[idx]]), axis=0, ddof=1)
        S1 = [np.mean(fB[idx] * (fAB[i][idx] - fA[idx]), axis=0) / V for i in range(k)]          # Saltelli 2010
        ST = [0.5 * np.mean((fA[idx] - fAB[i][idx]) ** 2, axis=0) / V for i in range(k)]          # Jansen
        return np.array(S1), np.array(ST)
    S1, ST = indices(np.arange(N))
    bs1, bst = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        a, b = indices(idx)
        bs1.append(a)
        bst.append(b)
    bs1, bst = np.array(bs1), np.array(bst)
    rows = []
    for i, name in enumerate(SOBOL_IN):
        for j, out in enumerate(SOBOL_OUT):
            rows.append({"input": name, "output": out, "S1": S1[i, j], "S1 lo": np.percentile(bs1[:, i, j], 2.5),
                         "S1 hi": np.percentile(bs1[:, i, j], 97.5), "ST": ST[i, j], "ST lo": np.percentile(bst[:, i, j], 2.5),
                         "ST hi": np.percentile(bst[:, i, j], 97.5), "N": N, "evaluations": N * (k + 2), "imputations per evaluation": SOBOL_M})
    return pd.DataFrame(rows)


# ── the full run ─────────────────────────────────────────────────────────────────────────────────
HEADLINE = ("mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)",
            "below the status tier (linked)")


def _save(df: pd.DataFrame, name: str) -> None:
    from atlas import logit_ci
    OUT.mkdir(parents=True, exist_ok=True)
    logit_ci.apply_frame(df, name).to_csv(OUT / name, index=False, float_format="%.6g")


def main(M: int = mi.M_DEFAULT) -> None:
    t0 = time.time()
    runs = {}
    for scen in mi.SCEN:
        runs[scen] = run_mi(scen, M, primary=(scen == "primary"))
        print(f"[{time.time() - t0:7.0f}s] MI {scen}: M = {M}", flush=True)
    acc_p, _, _ = runs["primary"]
    # national, cut and contrast tables
    nat = []
    for scen, (acc, _, _) in runs.items():
        for w in ("area", "energy", "count"):
            T = acc.table(("national", w))
            T.insert(0, "weight", w)
            T.insert(0, "scenario", scen)
            nat.append(T)
    _save(pd.concat(nat), "national.csv")
    cuts = []
    for scen, (acc, _, _) in runs.items():
        for g in acc.Q:
            if g[0] == "cut":
                T = acc.table(g)
                T.insert(0, "segment", g[3])
                T.insert(0, "cut", g[2])
                T.insert(0, "weight", g[1])
                T.insert(0, "scenario", scen)
                cuts.append(T)
    _save(pd.concat(cuts), "cuts.csv")
    one = []
    for scen, (acc, _, _) in runs.items():
        for g in (("h2b",), ("imputed-answer contrast",)) + ((("h4",), ("decomposition",)) if scen == "primary" else ()):
            T = acc.table(g)
            T.insert(0, "group", g[0])
            T.insert(0, "scenario", scen)
            one.append(T)
    _save(pd.concat(one), "reads_decomposition.csv")
    h5 = []
    for g in acc_p.Q:
        if g[0] in ("h5", "lever"):
            T = acc_p.table(g)
            T.insert(0, "item", g[1])
            T.insert(0, "group", g[0])
            h5.append(T)
    _save(pd.concat(h5), "h5_lever.csv")
    # M stability: T and B at M/2 against M for the headline statistics
    stab = []
    for scen, (acc, _, _) in runs.items():
        a, b = acc.table(("national", "area"), M // 2).set_index("statistic"), acc.table(("national", "area")).set_index("statistic")
        for s in HEADLINE:
            stab.append({"scenario": scen, "statistic": s, "T at M/2": a.loc[s, "T"], "T at M": b.loc[s, "T"],
                         "relative change in T": b.loc[s, "T"] / a.loc[s, "T"] - 1, "B at M/2": a.loc[s, "B"], "B at M": b.loc[s, "B"],
                         "relative change in B": b.loc[s, "B"] / a.loc[s, "B"] - 1 if a.loc[s, "B"] > 0 else np.nan,
                         "MC relative SE of B": np.sqrt(2 / (M - 1))})
    _save(pd.DataFrame(stab), "m_stability.csv")
    plog = []
    for scen, (_, _, log) in runs.items():
        tot = sum(log.values())
        for lvl, name in enumerate(("type x size x period", "type x size", "type", "all records")):
            plog.append({"scenario": scen, "donor cell": name, "redraws": log.get(lvl, 0), "share": log.get(lvl, 0) / tot if tot else np.nan})
    _save(pd.DataFrame(plog), "donor_cells.csv")
    print(f"[{time.time() - t0:7.0f}s] MI tables written", flush=True)
    w4 = float(acc_p.table(("h4",)).W.iloc[0])   # H4's within-imputation variance from the primary run
    from atlas import logit_ci
    # transfer tipping point
    deltas = np.round(np.arange(-3, 3.0001, 0.25), 2)
    tg, tt = [], []
    for variant in ("a", "b", "ab"):
        cf = [Cfg(da=float(x) if "a" in variant else 0.0, db=float(x) if "b" in variant else 0.0) for x in deltas]
        Gv = logit_ci.apply_frame(add_size_diffs(grid(M, cf, w4)), "transfer_grid.csv")
        Gv["variant"] = variant
        Gv["delta"] = Gv.da if variant != "b" else Gv.db
        tg.append(Gv)
        for r in tipping(Gv, "delta", {}):
            r["variant"] = variant
            tt.append(r)
    _save(pd.concat(tg), "transfer_grid.csv")
    _save(pd.DataFrame(tt), "transfer_tipping.csv")
    print(f"[{time.time() - t0:7.0f}s] transfer tipping written", flush=True)
    # model checks (the prior row is sensitivity S6 in atlas.sensitivities)
    det: list = []
    _save(model_checks(detail=det), "model_checks.csv")
    for name, T in model_check_partial(det).items():
        T.to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"[{time.time() - t0:7.0f}s] model checks written", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
