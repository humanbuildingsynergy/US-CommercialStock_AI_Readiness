"""Figure: who reaches AI readiness — mean linked level by size class, construction period, building type and the code
edition in force when each building was built (panel d replaces the former table of levels by edition).

Each row: the code-required to the declared level as a light bar, the linked level as a dot with its 95 % interval from
the multiple imputation (Rubin's rules); national mean dashed. Primary result, floor-area weighted
(results/estimates/cuts.csv). Labels carry each segment's share of national floor area
(results/estimates/zero_profile.csv, whole stock). Panel (d): results/editions/edition_levels.csv
(construction edition, primary scoring). No comparison markers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _common import COLORS, READING_COLORS, RES, Z, SMALL, figsize, plt, save

W6 = RES / "estimates"
k = pd.read_csv(W6 / "cuts.csv")
k = k[(k.scenario == "primary") & (k.weight == "area")]
est = k.pivot_table(index=["cut", "segment"], columns="statistic", values="estimate")
lo = k[k.statistic == "mean linked level"].set_index(["cut", "segment"]).lo95
hi = k[k.statistic == "mean linked level"].set_index(["cut", "segment"]).hi95
c = pd.DataFrame({"mean_C": est["mean linked level"], "mean_F": est["mean code-required level"],
                  "mean_D": est["mean declared level"], "lo": lo, "hi": hi}).reset_index()
zp = pd.read_csv(W6 / "zero_profile.csv")
zp = zp[zp.population == "whole stock"].rename(columns={"statistic": "segment", "estimate": "area_share"})
c = c.merge(zp[["cut", "segment", "area_share"]], on=["cut", "segment"], how="left")
n = pd.read_csv(W6 / "national.csv")
nat = float(n[(n.scenario == "primary") & (n.weight == "area") & (n.statistic == "mean linked level")].estimate.iloc[0])
col = READING_COLORS["frozen"]


def panel(ax, d, title, sort=False, xmax=0.47):
    if sort:
        d = d.sort_values("mean_C")
    y = np.arange(len(d))
    ax.hlines(y, d.mean_F, d.mean_D, color=col, lw=3.2, alpha=0.25)   # code-required to declared, as the legend says
    ax.errorbar(d.mean_C, y, xerr=[d.mean_C - d.lo, d.hi - d.mean_C], fmt="o", color=col, ms=3.2, elinewidth=0.8, capsize=1.5)
    ax.axvline(nat, color=COLORS["bwgray"], lw=0.6, ls=(0, (3, 2)))
    labels = [f"{s} ({100 * a:.0f}%)" if a >= 0.005 else f"{s} (<1%)" for s, a in zip(d.segment, d.area_share)]
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(0.0, xmax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.4)


ORDER = {"SIZE": ["<= 5,000", "5,001-24,999", ">= 25,000"],
         "YRCONC": ["Before 1946", "1946 to 1959", "1960 to 1969", "1970 to 1979", "1980 to 1989", "1990 to 1999",
                    "2000 to 2012", "2013 to 2018"]}
for cut, order in ORDER.items():
    rank = {seg: i for i, seg in enumerate(order)}
    c.loc[c.cut == cut, "rank"] = c.loc[c.cut == cut, "segment"].map(rank)
c = c.sort_values(["cut", "rank"], na_position="last")
size = c[c.cut == "SIZE"].copy()
size["segment"] = size.segment.str.replace("<= ", "≤ ").str.replace(">= ", "≥ ") + " ft²"
vint = c[c.cut == "YRCONC"].copy()
vint["segment"] = vint.segment.str.replace(" to ", "\u2013", regex=False)  # en-dash, as in the text and Figure 2
pba = c[c.cut == "PBA"].copy()
E = pd.read_csv(RES / "editions" / "edition_levels.csv")
E = E[(E.basis == "construction") & (E.scoring == "primary")]
ev = E.pivot(index="group", columns="statistic", values="estimate")
ER = ["before any 90.1-equivalent code", "90.1-1989", "90.1-1999/2001", "90.1-2004 and 2007", "90.1-2010", "90.1-2013 or later"]
ed = pd.DataFrame({"segment": ["Before 90.1-1989", "90.1-1989", "90.1-1999/2001", "90.1-2004, 2007", "90.1-2010", "90.1-2013 or later"],
                   "mean_C": ev.loc[ER, "mean linked level"].values, "mean_F": ev.loc[ER, "mean code-required level"].values,
                   "mean_D": ev.loc[ER, "mean declared level"].values, "area_share": ev.loc[ER, "share of floor area"].values,
                   "lo": E[E.statistic == "mean linked level"].set_index("group").lo95.loc[ER].values,
                   "hi": E[E.statistic == "mean linked level"].set_index("group").hi95.loc[ER].values})

fig = plt.figure(figsize=figsize("full", 1.12))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 8, 6], width_ratios=[1, 1.12], hspace=0.55, wspace=1.05)
a1 = fig.add_subplot(gs[0, 0])
a2 = fig.add_subplot(gs[1, 0])
a4 = fig.add_subplot(gs[2, 0])
a3 = fig.add_subplot(gs[:, 1])
panel(a1, size, "(a) Size class")
panel(a2, vint, "(b) Construction period")
panel(a3, pba, "(c) Building type", sort=True)
panel(a4, ed, "(d) Code edition in force at\nconstruction (wider axis)", xmax=0.65)
a4.set_xlabel("Mean AI readiness")
a3.set_xlabel("Mean AI readiness")
fig.legend(handles=[Line2D([], [], marker="o", ms=3.2, color=col, lw=0.8, label="linked level, 95% interval"),
                   Line2D([], [], color=col, lw=3.2, alpha=0.25, label="code-required to declared level"),
                   Line2D([], [], color=COLORS["bwgray"], lw=0.6, ls=(0, (3, 2)), label=f"national mean ({nat:.3f})"),
                   Line2D([], [], color="none", label="(%) share of national floor area")],
          loc="upper center", bbox_to_anchor=(0.5, 0.035),
          ncol=2, frameon=False, handlelength=1.6, columnspacing=2.0)
save(fig, "fig_segments")
