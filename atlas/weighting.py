"""How much the question set's weighting matters. The primary result (levels reached ÷ 76) does not change; the
sensitivities S19 (domain_equal) and S20 (intent_equal) are rows of atlas.sensitivities. This runner adds, per scoring, on the
primary imputations (M = 200, c = d = 1, Rubin's rules over each imputation's jackknife):

  size_ranking.csv     mean linked level by size class and the contrasts small − mid-size and mid-size − large
                       (replicate-wise differences, so the covariance of the two means is kept)
  loss_shares.csv      the four declared-to-linked losses by recording rule (Shapley over the four switches, as in
                       results/estimates/gap.csv), each as a share of the total loss
  group_shares.csv     the four stream-group Shapley shares of the linked level (as in estimates/decomposition_scenarios.csv),
                       each as a share of the mean
  tier_variant.csv     domain_equal_tiers: tier shares with each domain's question type weighted equally within an
                       intent, and the H2b ratio, beside the primary tiers
  canon_counts.csv     answer levels and question types by domain and by intent (from the canon)

    python -m atlas.weighting [M]
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd

from atlas import canon, config, draws, floors, logit_ci, mi, tiers
from atlas.mi import Cfg
from atlas.gap import SW_NAMES, SWITCHES

OUT = config.RESULTS / "weighting"
WORKERS = config.WORKERS
SCORINGS = ("primary", "domain_equal", "intent_equal")
SIZES = tuple(floors.SIZE_CLASSES)


def _rep(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Σ w·num / Σ w·den at the full weight and each replicate, (R+1,) per column of num."""
    return (w.T @ num) / (w.T @ den)[:, None]


def _qu(est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return est[0], ((est[1:] - est[0]) ** 2).sum(0)


def _job(m: int) -> tuple[int, list]:
    import warnings
    warnings.simplefilter("ignore")
    W = mi.weights()["area"]
    one = np.ones(W.shape[0])
    st = draws.stock()
    size = floors.size_class(st.df.SQFT).to_numpy()
    imp = mi.make_imp("primary", m)
    C = {}
    for S in itertools.product((False, True), repeat=4):
        C[S] = mi.evaluate(imp, Cfg("primary", **dict(zip(SWITCHES, S))))["C"]
    full = (True,) * 4
    out = []
    for sc in SCORINGS:
        val = {S: draws.score(c, sc)[:, 0] for S, c in C.items()}
        sC = draws.score(C[full], sc)
        # size ranking
        X = np.column_stack([sC[:, 0] * (size == s) for s in SIZES])
        D = np.column_stack([(size == s) * 1.0 for s in SIZES])
        means = (W.T @ X) / (W.T @ D)
        est = np.column_stack([means, means[:, 0] - means[:, 1], means[:, 1] - means[:, 2]])
        q, u = _qu(est)
        out.append((("size", sc), [*(f"mean linked, {s}" for s in SIZES), "size: small - mid", "size: mid - large"], q, u))
        # loss by recording rule (Shapley over the four switches), as shares of the total loss
        comp = []
        for i in range(4):
            tot = 0.0
            for S in itertools.product((False, True), repeat=4):
                if S[i]:
                    continue
                k = sum(S)
                Sn = tuple(True if j == i else S[j] for j in range(4))
                tot = tot + math.factorial(k) * math.factorial(3 - k) / math.factorial(4) * (val[Sn] - val[S])
            comp.append(-tot)                                   # loss = minus the drop
        comp = np.column_stack(comp)
        total = comp.sum(1)
        est = np.column_stack([_rep(comp, total, W), (W.T @ comp) / W.sum(0)[:, None], (W.T @ total) / W.sum(0)])
        q, u = _qu(est)
        out.append((("loss", sc), [*(f"share of loss: {SW_NAMES[s]}" for s in SWITCHES),
                                   *(f"loss: {SW_NAMES[s]}" for s in SWITCHES), "loss: total (declared minus linked)"], q, u))
        # stream-group shares of the linked level
        ph = draws.phi(C[full], sc)
        est = np.column_stack([_rep(ph, sC[:, 0], W), (W.T @ ph) / W.sum(0)[:, None], (W.T @ sC[:, 0]) / W.sum(0)])
        q, u = _qu(est)
        out.append((("groups", sc), [*(f"share of mean: {g}" for g in draws.GROUP_NAMES),
                                     *(f"linked {g}" for g in draws.GROUP_NAMES), "mean linked level"], q, u))
    # tier variant (domain_equal_tiers) beside the primary tiers
    th = tiers.THRESHOLD - 1e-12
    Wc = mi.weights()["count"]
    for sc in ("primary", "domain_equal_tiers"):
        sC = draws.score(C[full], sc)
        cols = {"status tier (linked)": (sC[:, 1] >= th) * 1.0, "summary tier (linked)": (sC[:, 2] >= th) * 1.0,
                "diagnosis tier (linked)": (sC[:, 3] >= th) * 1.0, "below the status tier (linked)": (sC[:, 1] < th) * 1.0}
        q, u, names = mi.jackknife(cols, W)
        s = cols["status tier (linked)"]
        v = ((W.T @ s) / W.sum(0)) / np.maximum((Wc.T @ s) / Wc.sum(0), 1e-12)
        out.append((("tiers", sc), [*names, "H2b concentration ratio"], [*q, v[0]], [*u, ((v[1:] - v[0]) ** 2).sum()]))
    return m, out


def run_jobs(M: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    acc: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (m, out) in enumerate(ex.map(_job, range(M), chunksize=1), 1):
            for g, names, Q, U in out:
                a = acc.setdefault(g, {"names": list(names), "Q": [], "U": []})
                a["Q"].append(np.asarray(Q, float))
                a["U"].append(np.asarray(U, float))
            if k % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] imputation {k} of {M}", flush=True)
    tabs = {}
    for g, a in acc.items():
        T = mi.rubin(np.vstack(a["Q"]), np.vstack(a["U"]))
        T.insert(0, "statistic", a["names"])
        T.insert(0, "scoring", g[1])
        tabs.setdefault(g[0], []).append(T)
    return {k: pd.concat(v, ignore_index=True) for k, v in tabs.items()}


def canon_counts() -> pd.DataFrame:
    c = canon.CANONS["v2"]
    rows = []
    for part, keys in (("domain", canon.DOMAINS), ("intent", canon.INTENTS)):
        j = 0 if part == "domain" else 1
        for k in keys:
            cells = [cell for cell in c.rungs if cell[j] == k]
            rows.append({"by": part, "name": k, "answer levels": sum(len(c.rungs[x]) for x in cells), "question types": len(cells),
                         "weight under primary": sum(len(c.rungs[x]) for x in cells) / c.total, "weight under equal": 1 / len(keys)})
    return pd.DataFrame(rows)


def main(M: int = mi.M_DEFAULT) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    canon_counts().to_csv(OUT / "canon_counts.csv", index=False, float_format="%.6g")
    tabs = run_jobs(M)
    for key, name in (("size", "size_ranking.csv"), ("loss", "loss_shares.csv"), ("groups", "group_shares.csv"),
                      ("tiers", "tier_variant.csv")):
        T = tabs[key]
        T = logit_ci.apply_frame(T, name) if key == "tiers" else T
        T.to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"[{time.time() - t0:6.0f}s] wrote results/weighting/", flush=True)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
