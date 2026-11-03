"""Figure: the whole distribution — share of floor area whose linked level is at or above x, for the
primary result and the strict and enriched scenarios, with Rubin's-rules 95 % bands (multiple imputation, M = 200).
Source: results/estimates/national.csv (statistics 'linked at or above x', area weight)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from matplotlib.patches import Patch

from _common import COLORS, READING_COLORS, RES, SMALL, figsize, plt, save

N = pd.read_csv(RES / "estimates" / "national.csv")
N = N[(N.weight == "area") & N.statistic.str.startswith("linked at or above")].copy()
N["x"] = N.statistic.str.extract(r"([\d.]+)$").astype(float)
fig, ax = plt.subplots(figsize=figsize("column", 0.82))
for scen, key, lab in (("strict (low)", "strict", "Strict (low)"), ("primary", "frozen", "Primary"), ("enriched (high)", "enriched", "Enriched (high)")):
    s = N[N.scenario == scen].sort_values("x")
    col = READING_COLORS[key]
    ax.fill_between(s.x, 100 * s.lo95.clip(0, 1), 100 * s.hi95.clip(0, 1), color=col, alpha=0.18, lw=0, step="post")
    ax.step(s.x, 100 * s.estimate, where="post", color=col, lw=1.1, label=lab)
ax.axvline(0.5, color=COLORS["bwgray"], lw=0.7, ls=(0, (3, 2)))
ax.text(0.505, 50, "x = 0.5", fontsize=SMALL, color=COLORS["bwgray"])
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 101)
ax.set_xlabel("Linked level x (AI readiness)")
ax.set_ylabel("Floor area at or above x (%)")
ax.grid(alpha=0.4)
h, l = ax.get_legend_handles_labels()
ax.legend(h + [Patch(color=COLORS["bwgray"], alpha=0.3, lw=0)], l + ["95% interval"], loc="upper right", fontsize=SMALL)
save(fig, "fig_distribution")
