"""Figure: the reference-set-to-stock transfer — each headline conclusion's statistic against the reference-set shift δ
(the log-odds of linkage type and class rates shifted together), with Rubin's-rules 95 % intervals and, dashed, the
value at which the conclusion would reverse. The declared level, flat in δ, has no panel; the small < mid-size
ranking (the quantity the text cites) has one. Source: results/estimates/transfer_grid.csv (variant 'ab')."""
from __future__ import annotations

import pandas as pd

from _common import COLORS, READING_COLORS, RES, SMALL, figsize, note, plt, save

G = pd.read_csv(RES / "estimates" / "transfer_grid.csv")
G = G[G.variant == "ab"]
REVERSAL = COLORS["bw4"]          # one colour for every reversal line
panels = [("below the status tier (linked)", 0.5, "(a) Below the status tier (H1)", 100, "Floor area (%)"),
          ("H2b", 1.0, "(b) Status-tier concentration (H2)", 1, "Floor-area ÷ building share"),
          ("H4", 0.0, "(c) Construction period − building type (H3)", 1, "Difference in R² share"),
          ("size: small - mid", 0.0, "(d) Small − mid-size buildings (H2)", 1, "Difference in linked level"),
          ("size: mid - large", 0.0, "(e) Mid-size − large buildings (H2)", 1, "Difference in linked level")]
fig, axs = plt.subplots(3, 2, figsize=figsize("full", 1.0))
axes = axs.ravel()
for ax, (stat, bound, title, scale, ylab) in zip(axes, panels):
    s = G[G.statistic == stat].sort_values("delta")
    col = READING_COLORS["frozen"]
    ax.fill_between(s.delta, scale * s.lo95, scale * s.hi95, color=col, alpha=0.18, lw=0)
    ax.plot(s.delta, scale * s.estimate, color=col, lw=1.1)
    ax.axhline(scale * bound, color=REVERSAL, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(0, color=COLORS["bwgray"], lw=0.5)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Reference-set shift δ (log-odds)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.4)
key = axs[2, 1]
key.axis("off")
key.plot([], [], color=READING_COLORS["frozen"], lw=1.1, label="Estimate")
key.fill_between([], [], [], color=READING_COLORS["frozen"], alpha=0.18, lw=0, label="95% interval")
key.plot([], [], color=REVERSAL, lw=0.8, ls=(0, (3, 2)), label="Where the conclusion would reverse")
key.legend(loc="upper left", frameon=False)
key.text(0.0, 0.40, "δ > 0: buildings with a BAS record and link\ntheir data at $e^{δ}$ times the reference-set odds.\n"
         "δ < 0: less often. δ = 0 is the primary result.", transform=key.transAxes, fontsize=SMALL, color="#333333", va="top")
fig.tight_layout(w_pad=1.5, h_pad=1.2)
save(fig, "fig_tipping")
