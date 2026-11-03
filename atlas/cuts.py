"""Mean linked level and tier shares by census division, climate and ownership under the floor-area, energy and
building-count weights, from the primary imputations (M = 200, Rubin's rules). ``atlas.estimates`` computes the energy
and count weights for size, period and type only. Size, type and period are repeated here for the share of floor area
with no linked data (zero linked level), which its cuts.csv does not carry.

    python -m atlas.cuts [M]   -> results/estimates/cuts_weights.csv
"""
from __future__ import annotations

import pandas as pd

from atlas import draws, mi, stock as stk
from atlas.mi import Cfg
from atlas.estimates import Acc

STATS = ("mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)", "linked level zero")


def main(M: int = mi.M_DEFAULT) -> None:
    st = draws.stock()
    W = mi.weights()
    labs = {v: stk.cut_labels(st.df, v) for v in ("CENDIV", "PUBCLIM", "OWNTYPE", "OWNOCC", "SIZE", "PBA", "YRCONC")}
    acc = Acc()
    for m in range(M):
        ev = mi.evaluate(mi.make_imp("primary", m), Cfg("primary"))
        cols = {k: v for k, v in mi.columns(ev, full=False).items() if k in STATS}
        for var, (lab, order) in labs.items():
            for g in order:
                mask = (lab == g).to_numpy()
                for w in ("area", "energy", "count"):
                    if W[w][mask, 0].sum() <= 0:
                        continue
                    Q, U, names = mi.jackknife(cols, W[w], mask)
                    acc.add((var, g, w), names, Q, U)
    out = []
    for key in acc.Q:
        T = acc.table(key)
        T.insert(0, "weight", key[2])
        T.insert(0, "segment", key[1])
        T.insert(0, "cut", key[0])
        out.append(T)
    from atlas import logit_ci
    logit_ci.apply_frame(pd.concat(out, ignore_index=True), "cuts_weights.csv").to_csv(mi.OUT / "cuts_weights.csv", index=False,
                                                                                       float_format="%.6g")
    print("wrote cuts_weights.csv")


if __name__ == "__main__":
    import sys
    import warnings
    warnings.simplefilter("ignore")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else mi.M_DEFAULT)
