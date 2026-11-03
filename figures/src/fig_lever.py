"""Figure: what added data would do (exploratory), primary result only. Gain in the floor-area share at the
summary and diagnosis tiers, percentage points: (a) the 90.1-2013 §8.4.3 monitoring bundle (total electricity logged with
it) added where the clause applies (>= 25,000 ft²) or to every building, with and without reading its HVAC monitoring as
equipment input power, labelled with the gain in the mean linked level; (b) single data streams added to every building as
recorded, linked data (formerly appendix Table F2), those that move either tier by at least half a point. Multiple
imputation (M = 200), Rubin's-rules 95 % intervals. Source: results/estimates/h5_lever.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _common import TIER_COLORS, RES, SMALL, figsize, plt, save

H = pd.read_csv(RES / "estimates" / "h5_lever.csv").set_index(["group", "item", "statistic"])
g = lambda grp, k, s, c="estimate": float(H.loc[(grp, k, s), c])
# "where §8.4.3 applies" = at least 25,000 ft2 and not a hotel, motel or restaurant (exception f); 62% of floor area
bundle = [("bundle, >= 25,000 ft2 only", "Where §8.4.3 applies (62% of floor area)"),
          ("bundle without equipment power, >= 25,000 ft2 only", "Where §8.4.3 applies, no equipment power"),
          ("bundle, every building", "Every building"),
          ("bundle without equipment power, every building", "Every building, no equipment power")]
NAMES = {"Occ": "Occupancy", "Pwr": "Equipment power", "CO2": "Zone CO$_2$", "Lgt": "Lighting", "Sub": "End-use submeter",
         "Whole": "Whole-building interval meter"}
singles = [k for k in H.loc["h5"].index.get_level_values(0).unique() if k in NAMES
           and max(g("h5", k, "summary gain"), g("h5", k, "diagnosis gain")) >= 0.005]
singles.sort(key=lambda k: (g("h5", k, "summary gain"), g("h5", k, "diagnosis gain")), reverse=True)

fig, (ax, bx) = plt.subplots(2, 1, figsize=figsize("full", 0.86), gridspec_kw={"height_ratios": [4, 5]})
w = 0.36


def bars(a, grp, items, xmax, step):
    y = np.arange(len(items))[::-1]
    for kk, tier in enumerate(("summary", "diagnosis")):
        off = (0.5 - kk) * w
        for yi, k in zip(y, items):
            e, lo, hi = (100 * g(grp, k, f"{tier} gain", c) for c in ("estimate", "lo95", "hi95"))
            a.barh(yi + off, e, w, color=TIER_COLORS[tier], label=tier.capitalize() if yi == y[0] else None,
                   xerr=[[max(e - lo, 0)], [max(hi - e, 0)]], error_kw={"elinewidth": 0.7, "capsize": 1.2})
            a.text(max(hi, e) + step, yi + off, f"+{e:.1f}", va="center", fontsize=SMALL)
    a.set_xlim(0, xmax)
    a.set_ylim(y.min() - 0.6, y.max() + 0.6)
    a.grid(axis="x", alpha=0.4)
    a.set_xlabel("Gain in the stock's floor area at the tier (percentage points)")
    return y


y = bars(ax, "lever", [k for k, _ in bundle], 104, 1.5)
ax.set_yticks(y, [f"{lab}\nmean linked level +{g('lever', k, 'mean linked gain'):.3f}" for k, lab in bundle], fontsize=SMALL)
ax.set_title("(a) The 90.1-2013 monitoring streams, added as linked data", loc="left")

y = bars(bx, "h5", singles, 7.5, 0.12)
bx.set_yticks(y, [NAMES[k] for k in singles])
bx.set_title("(b) One stream added to every building (streams moving a tier by 0.5 point or more)", loc="left")
h, l = bx.get_legend_handles_labels()
bx.legend(h + [Line2D([], [], color="black", lw=0.7, marker="|", ms=4)], l + ["95% interval"], loc="lower right", frameon=False)
fig.tight_layout(h_pad=1.4)
save(fig, "fig_lever")
