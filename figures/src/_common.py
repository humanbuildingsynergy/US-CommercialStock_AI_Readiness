"""Shared helpers for the figure scripts (figures/src/<id>.py).

Every script reads the result tables only (results/**.csv), never recomputes, and writes figures/<id>.pdf plus
figures/_png/<id>.png. Style: Okabe-Ito palette, print sizes at the paper's text width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "figures"
Z = 1.959964  # 95 % normal quantile, as in the ledger

COLORS = {"bw1": "#0072B2", "bw2": "#E69F00", "bw3": "#009E73", "bw4": "#D55E00", "bw5": "#CC79A7", "bw6": "#56B4E9",
          "bw7": "#F0E442", "bwgray": "#7F7F7F", "bwlight": "#E6E6E6", "bwtext": "#222222"}
CYCLE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
                     "axes.linewidth": 0.6, "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "figure.dpi": 150})
from cycler import cycler  # noqa: E402

plt.rcParams["axes.prop_cycle"] = cycler(color=CYCLE)

# House font: Helvetica for every figure. Only the regular face of the macOS font
# collection is visible to matplotlib, so figures set hierarchy by size and shade, not weight.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica"})

# Figures are drawn at the manuscript's text width, so LaTeX includes them at 100 % and every font prints at its set
# size; nothing prints below 7 pt.
TEXT_W = 390.0 / 72.27   # \textwidth of the review layout, inches
SMALL = 7.0              # smallest print size: ticks, in-plot labels, legends, notes


def figsize(width="column", aspect=0.62):
    w = {"column": 0.6 * TEXT_W, "full": TEXT_W}[width]
    return (w, w * aspect)


plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 8.0, "axes.labelsize": 7.5, "xtick.labelsize": SMALL,
                     "ytick.labelsize": SMALL, "legend.fontsize": SMALL, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# Gridlines sit behind bars, markers and text in every figure.
plt.rcParams.update({"axes.axisbelow": True})

# One colour per reading, used in every figure that shows readings.
READING_COLORS = {
    "strict": COLORS["bwgray"],
    "census": COLORS["bw6"],
    "frozen": COLORS["bw1"],
    "enriched": COLORS["bw2"],
}
TIER_COLORS = {"status": COLORS["bw1"], "summary": COLORS["bw3"], "diagnosis": COLORS["bw4"]}


def est_se(cell: str) -> tuple[float, float]:
    """'est|se' cells written by atlas.bounds."""
    e, s = map(float, str(cell).split("|"))
    return e, s


def _labels_over_grid(fig) -> None:
    """Give in-plot text labels a white backing so gridlines run behind them. Labels that already
    carry a box are left alone."""
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_bbox_patch() is None:
                t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))
            t.set_zorder(max(t.get_zorder(), 5))


def note(fig, text: str, y: float = -0.02) -> None:
    """Figure note under the axes, wrapped to the figure's width so it never widens the saved figure."""
    import textwrap
    width = int(fig.get_figwidth() * 72 / (0.5 * SMALL))   # characters per line at SMALL pt
    fig.text(0.01, y, "\n".join(textwrap.wrap(" ".join(text.split()), width)), ha="left", va="top", fontsize=SMALL,
             color="#333333", linespacing=1.35)


def save(fig, fid: str) -> None:
    _labels_over_grid(fig)
    w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * plt.rcParams["savefig.pad_inches"]
    if w > 1.02 * TEXT_W:
        print(f"warning: {fid} is {w:.2f} in wide, {w / TEXT_W:.0%} of the text width; LaTeX will shrink it")
    (OUT / "_png").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fid}.pdf", metadata={"CreationDate": None})   # no timestamp: the file reproduces byte for byte
    fig.savefig(OUT / "_png" / f"{fid}.png", dpi=300)
    plt.close(fig)
    print(f"wrote figures/{fid}.pdf")
