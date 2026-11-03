"""Figure: global sensitivity over the scenario dimensions and the reference-set shift — first-order and total-effect
Sobol indices of compliance c, description d and the reference-set shift δ, for the mean linked level and each tier
share, with bootstrap 95 % intervals, on two range sets: (a) c and d in [0.5, 1]; (b) c in [0.5, 1] and d over
its full range [0.05, 1]; δ in [-3, 3] in both. The indices apportion variance over these ranges only. d applies
to data recorded outside the BAS only, one draw per stream (the design).
Source: results/scenarios/sobol_a_stream.csv, sobol_b_stream.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import COLORS, RES, SMALL, figsize, note, plt, save

SETS = [("(a) c, d in [0.5, 1]", pd.read_csv(RES / "scenarios" / "sobol_a_stream.csv")),
        ("(b) d in [0.05, 1]", pd.read_csv(RES / "scenarios" / "sobol_b_stream.csv"))]
outs = ["mean linked level", "status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)"]
labels = {"mean linked level": "Mean linked level", "status tier (linked)": "Status tier", "summary tier (linked)": "Summary tier",
          "diagnosis tier (linked)": "Diagnosis tier"}
inputs = [("c", "compliance c"), ("d", "description d"), ("delta", "reference-set shift δ")]
cols = {"c": COLORS["bw2"], "d": COLORS["bw3"], "delta": COLORS["bw1"]}
fig, axes = plt.subplots(2, len(outs), figsize=figsize("full", 0.68), sharey=True)
for r, (title, S) in enumerate(SETS):
    for j, (ax, o) in enumerate(zip(axes[r], outs)):
        s = S[S.output == o].set_index("input")
        x = np.arange(len(inputs))
        for i, (k, lab) in enumerate(inputs):
            ax.bar(i - 0.18, s.loc[k, "S1"], 0.34, color=cols[k], alpha=0.55,
                   yerr=[[max(s.loc[k, "S1"] - s.loc[k, "S1 lo"], 0)], [max(s.loc[k, "S1 hi"] - s.loc[k, "S1"], 0)]],
                   error_kw={"elinewidth": 0.6, "capsize": 1})
            ax.bar(i + 0.18, s.loc[k, "ST"], 0.34, color=cols[k],
                   yerr=[[max(s.loc[k, "ST"] - s.loc[k, "ST lo"], 0)], [max(s.loc[k, "ST hi"] - s.loc[k, "ST"], 0)]],
                   error_kw={"elinewidth": 0.6, "capsize": 1})
        ax.set_xticks(x, ["c", "d", "δ"])
        ax.text(0, 0.03, "≈ 0", ha="center", fontsize=SMALL, color="#555555")
        ax.set_title(labels[o], loc="left", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.4)
    axes[r][0].set_ylabel(f"{title}\nSobol index", fontsize=SMALL)
from matplotlib.patches import Patch
# the legend shows the bars as drawn: each input's colour, lighter (left) = first-order, darker (right) = total-effect
fig.legend(handles=[h for k, lab in inputs for h in (Patch(color=cols[k], alpha=0.55, label=f"{lab}, first order"),
                                                     Patch(color=cols[k], label=f"{lab}, total effect"))],
           loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=3, frameon=False, columnspacing=1.2)
fig.tight_layout(w_pad=0.6, h_pad=1.6)
save(fig, "fig_sobol")
