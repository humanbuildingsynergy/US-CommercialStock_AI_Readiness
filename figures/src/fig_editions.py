"""Figure 2: the code edition in force, from the imputed editions. Floor-area share of the stock under
each 90.1-equivalent construction edition, by census division (a) and by construction period (b), averaged over the
M = 200 imputations (state ∝ commercial energy use within the division, year uniform within the period). The 2013
edition, the first with building-level monitoring, is outlined. Source: results/estimates/editions_imputed.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from _common import COLORS, RES, SMALL, figsize, note, plt, save

E = pd.read_csv(RES / "estimates" / "editions_imputed.csv")
LADDER = ["pre-1989", "1989", "1999/2001", "2004", "2007", "2010", "2013", "2016"]
LABEL = {"pre-1989": "before 90.1-1989"}
# a stepped palette whose neighbours are distinguishable and whose oldest edition is not near-white
shades = ["#B9B9B9", "#8C6D46", "#D29A4C", "#E8C66A", "#9CC5A1", "#5AA6A0", "#2F6FAE", "#1B3A6B"]
DIVS = {"New England": "New\nEngland", "Middle Atlantic": "Middle\nAtlantic", "East North Central": "East N.\nCentral",
        "West North Central": "West N.\nCentral", "South Atlantic": "South\nAtlantic", "East South Central": "East S.\nCentral",
        "West South Central": "West S.\nCentral", "Mountain": "Mountain", "Pacific": "Pacific"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize("full", 0.60), gridspec_kw={"width_ratios": [1.15, 1]})
for a, cut, title in ((ax, "CENDIV", "(a) By census division"), (bx, "YRCONC", "(b) By construction period")):
    sub = E[E.cut == cut]
    segs = list(dict.fromkeys(sub.segment))
    x = np.arange(len(segs))
    bottom = np.zeros(len(segs))
    for k, ed in enumerate(LADDER):
        vals = np.array([100 * float(sub[(sub.segment == s) & (sub.statistic == ed)].estimate.iloc[0]) if ed in set(sub.statistic) else 0.0
                         for s in segs])
        a.bar(x, vals, 0.72, bottom=bottom, color=shades[k], edgecolor="#1A1A1A" if ed == "2013" else "#BBBBBB",
              linewidth=1.0 if ed == "2013" else 0.3)
        bottom += vals
    labels = [DIVS.get(s, s) for s in segs] if cut == "CENDIV" else [s.replace(" to ", "–") for s in segs]
    labels = [lab.replace("\n", " ") for lab in labels]
    a.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
    a.set_ylim(0, 100)
    a.set_title(title, loc="left", fontsize=7)
    a.grid(axis="y", alpha=0.4)
ax.set_ylabel("Floor area under the edition (%)")
fig.legend(handles=[Patch(facecolor=c, edgecolor=("#1A1A1A" if l == "2013" else "#BBBBBB"), lw=(1.0 if l == "2013" else 0.3),
                          label=("2013 (monitoring clause)" if l == "2013" else LABEL.get(l, l))) for l, c in zip(LADDER, shades)],
           loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=4, frameon=False, handlelength=1.2,
           title=f"90.1-equivalent construction edition (imputed, mean over {int(E.M.iloc[0])} imputations)", title_fontsize=SMALL)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_editions")
