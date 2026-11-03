"""The reference set's stream counts on one stated basis.

The reference set is the readiness framework's 45 knowledge graphs. Three nested populations:
  all 45 graphs;
  the 37 whose declared streams answer at least one question (C_declared > 0) -- the linkage model's population;
  the 6 of those that link part of their declared data ('partial'), whose per-stream linked shares are the class rates
  of a partly linked building (the Method's definition, which the code follows).
Linked = the stream carries a time-series reference in the published graph. A graph's type (all / none / partial) is
defined on answerability (C_linked / C_declared), not on stream counts.

    python -m atlas.refset_counts   -> results/scoring/reference_set_counts.csv
"""
from __future__ import annotations

import pandas as pd

from atlas import config, canon, linkage

OUT = config.RESULTS / "scoring"


def main() -> pd.DataFrame:
    g = linkage.graphs()
    pops = {"all 45": g, "37 with C_declared > 0": g[g.in_R1], "6 partial": g[g.type == "partial"]}
    rows = []
    for c in canon.STREAMS:
        r = {"stream": c, "meaning": canon.STREAMS[c]}
        for name, p in pops.items():
            r[f"declared, {name}"] = int(p.declared.map(lambda s: c in s).sum())
            r[f"linked, {name}"] = int(p.apply(lambda x: c in x.declared and c in x.linked, axis=1).sum())
        rows.append(r)
    for combo in (("Run", "Load"),):
        r = {"stream": "+".join(combo), "meaning": "both streams declared (joint count)"}
        for name, p in pops.items():
            r[f"declared, {name}"] = int(p.declared.map(lambda s: set(combo) <= s).sum())
            r[f"linked, {name}"] = int(p.apply(lambda x: set(combo) <= (x.declared & x.linked), axis=1).sum())
        rows.append(r)
    D = pd.DataFrame(rows)
    D["partial rate used (point)"] = [
        (r["linked, 6 partial"] / r["declared, 6 partial"]) if r["declared, 6 partial"] >= linkage.MIN_DECL else
        (1 + r["linked, 6 partial"]) / (2 + r["declared, 6 partial"]) for _, r in D.iterrows()]
    D["rate basis"] = ["empirical (>= 5 partial declarations)" if r["declared, 6 partial"] >= linkage.MIN_DECL else
                       "Beta(1, 1) posterior mean (< 5 partial declarations)" for _, r in D.iterrows()]
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "reference_set_counts.csv", index=False, float_format="%.4g")
    return D


if __name__ == "__main__":
    print(main().drop(columns="meaning").to_string())
