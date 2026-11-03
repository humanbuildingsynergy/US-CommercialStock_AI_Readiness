"""Figure: the data gap. (a) Horizontal waterfall of the stock mean: code-required level → declared level →
linked level, the declared-to-linked drop split exactly (Shapley) over the thermostat rule, the meter rule, the BAS and
network route rule and the BAS linkage draw (description d = 1 adds nothing). (b) The same three levels divided among four
groups of data streams (Shapley; replaces the former decomposition table). Multiple imputation (M = 200), Rubin's-rules
95 % intervals. Sources: results/estimates/gap.csv (area weight), results/estimates/decomposition_scenarios.csv."""
from __future__ import annotations

import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _common import COLORS, READING_COLORS, RES, SMALL, figsize, plt, save

G = pd.read_csv(RES / "estimates" / "gap.csv")
G = G[G.weight == "area"].set_index("statistic")
v = lambda k, c="estimate": float(G.loc[k, c])
DS = pd.read_csv(RES / "estimates" / "decomposition_scenarios.csv").set_index(["scenario", "statistic"])

# (label, statistic, kind): level bars start at 0, steps start where the previous bar ended
steps = [("Code-required level", "code-required level", "level"),
         ("Reported controls add their streams", "declared minus code-required", "up"),
         ("Declared level", "declared level", "level"),
         ("Conventional thermostats keep no record", "drop: thermostat rule", "down"),
         ("Utility meter gives no interval readings", "drop: meter rule", "down"),
         ("No automation system records the sensor", "drop: BAS and network route rule", "down"),
         ("BAS points not linked to recorded series", "drop: BAS linkage draw", "down"),
         ("Linked level", "linked level", "level")]
fig, (ax, bx) = plt.subplots(2, 1, figsize=figsize("full", 0.82), sharex=True, gridspec_kw={"height_ratios": [8, 3.6]})
run = 0.0
for i, (lab, k, kind) in enumerate(steps):
    val, lo, hi = v(k), v(k, "lo95"), v(k, "hi95")
    if kind == "level":
        ax.barh(i, val, 0.62, color=READING_COLORS["frozen"] if lab.startswith("Linked") else COLORS["bwgray"], alpha=0.9)
        ax.errorbar(val, i, xerr=[[val - lo], [hi - val]], color="#333333", elinewidth=0.7, capsize=1.5)
        ax.text(hi + 0.006, i, f"{val:.3f}", va="center", fontsize=SMALL)
        run = val
    else:
        left = run if kind == "up" else run + val
        ax.barh(i, abs(val), 0.62, left=left, color=COLORS["bw6"] if kind == "up" else COLORS["bw4"], alpha=0.85)
        ax.text(max(run, run + val) + 0.006, i, f"{val:+.3f}".replace("-", "−"), va="center", fontsize=SMALL)
        run = run + val
    if i < len(steps) - 1:
        ax.plot([run, run], [i + 0.31, i + 0.69], color="#555555", lw=0.6, ls=(0, (2, 1.5)))   # next bar starts here
d0, d1 = 3, 6   # the four recording-rule rows
drop = v("linked level") - v("declared level")
xb = 0.392
ax.plot([xb - 0.004, xb, xb, xb - 0.004], [d0 - 0.3, d0 - 0.3, d1 + 0.3, d1 + 0.3], color="#333333", lw=0.7)
ax.text(xb + 0.005, (d0 + d1) / 2, f"{drop:+.3f}".replace("-", "−") + "\nin all\n(Shapley split)", va="center", fontsize=SMALL)
ax.set_yticks(range(len(steps)), [s[0] for s in steps], fontsize=SMALL)
ax.invert_yaxis()
ax.set_xlim(0, 0.42)
ax.tick_params(labelbottom=True)
ax.set_title("(a) From the code-required to the linked level", loc="left")

ax.grid(axis="x", alpha=0.4)

groups = [("thermostat", "Thermostat", COLORS["bw1"]), ("metering", "Metering", COLORS["bw2"]),
          ("BAS HVAC points", "BAS HVAC points", COLORS["bw5"]), ("occupancy, lighting, CO2", "Occupancy, lighting, CO$_2$", COLORS["bw3"])]
levels = [("code-required", "Code-required level"), ("declared", "Declared level"), ("linked", "Linked level")]
for j, (lvl, _) in enumerate(levels):
    left = 0.0
    for g, glab, col in groups:
        x = float(DS.loc[("primary", f"{lvl} {g}"), "estimate"])
        bx.barh(j, x, 0.62, left=left, color=col, alpha=0.85, label=glab if j == 0 else None)
        if x >= 0.03:
            bx.text(left + x / 2, j, f"{x:.3f}", ha="center", va="center", fontsize=SMALL, color="white",
                    bbox=dict(facecolor="none", edgecolor="none"))   # inside the bar: no white backing
        left += x
    bx.text(left + 0.006, j, f"{left:.3f}", va="center", fontsize=SMALL)
bx.set_yticks(range(len(levels)), [l[1] for l in levels])
bx.invert_yaxis()
bx.set_xlabel("Mean AI readiness (share of the 76 answer levels)")
bx.set_title("(b) What each level is made of (Shapley split)", loc="left")
bx.grid(axis="x", alpha=0.4)
h, l = bx.get_legend_handles_labels()
fig.tight_layout(h_pad=1.2)
KEY = dict(loc="upper center", ncol=4, frameon=False, columnspacing=1.0, handlelength=1.0)
fig.legend(handles=[Patch(color=COLORS["bwgray"], alpha=0.9, label="(a) Level, from zero"),
                   Patch(color=COLORS["bw6"], alpha=0.85, label="Added to the bar above"),
                   Patch(color=COLORS["bw4"], alpha=0.85, label="Removed from the bar above"),
                   Line2D([], [], color="#333333", lw=0.7, marker="|", ms=4, label="95% interval")],
           bbox_to_anchor=(0.5, 0.0), **KEY)
fig.legend(h, ["(b) " + l[0], *l[1:]], bbox_to_anchor=(0.5, -0.045), **KEY)
save(fig, "fig_gap")
