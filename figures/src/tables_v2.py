"""Results tables, from the multiple imputation (results/estimates/):

  table_headline      strict / primary / enriched: code-required, declared and linked level, tier shares, FMI, and the
                      linked-level mean under energy and building-count weights, and the estimates that answer H1–H3
  table_zero_profile  who has no linked data: distribution of that floor area over size, type, period and division,
                      and its BAS, smart-thermostat and interval-meter shares, beside the whole stock
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import tables as T1   # Table 1's writer (figures/src/tables.py) — importing it also rewrites Table 1

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "estimates"
write = T1.write
N = pd.read_csv(D / "national.csv").set_index(["scenario", "weight", "statistic"])
SC = [("strict (low)", "Strict (low)"), ("primary", "Primary"), ("enriched (high)", "Enriched (high)")]
M = int(N["M"].max())   # imputations per scenario, read from the results
MI = f"multiple imputation, M = {M}, Rubin's rules"


def ci(s, stat, w="area", pct=True):
    r = N.loc[(s, w, stat)]
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(r.estimate)}\n[{f(r.lo95)}, {f(r.hi95)}]"


# ---------------------------------------------------------------- headline
rows = []
for stat, lab, pct in (("mean code-required level", "Code-required level (mean)", False), ("mean declared level", "Declared level (mean)", False),
                       ("mean linked level", "Linked level (mean)", False), ("status tier (linked)", "Status tier (% of floor area)", True),
                       ("summary tier (linked)", "Summary tier (%)", True), ("diagnosis tier (linked)", "Diagnosis tier (%)", True),
                       ("linked level zero", "No linked data (%)", True)):
    rows.append([lab, *[ci(s, stat, pct=pct) for s, _ in SC], f"{N.loc[('primary', 'area', stat), 'fmi']:.2f}"])
rows.append(["Linked level (mean), energy-weighted", *[ci(s, "mean linked level", "energy", False) for s, _ in SC], f"{N.loc[('primary', 'energy', 'mean linked level'), 'fmi']:.2f}"])
rows.append(["Linked level (mean), building-weighted", *[ci(s, "mean linked level", "count", False) for s, _ in SC], f"{N.loc[('primary', 'count', 'mean linked level'), 'fmi']:.2f}"])
# the estimates that answer the hypotheses (formerly their own table)
RD = pd.read_csv(D / "reads_decomposition.csv")
RD = RD[RD.group.isin(["h2b", "h4"])].set_index(["scenario", "group"])   # one row per scenario and estimate
rd = lambda s, g, f: (lambda r: f"{f(r.estimate)}\n[{f(r.lo95)}, {f(r.hi95)}]")(RD.loc[(s, g)]) if (s, g) in RD.index else "—"
sg = lambda t: t.replace("-", "−")   # true minus signs
rows.append([r"\textit{Hypotheses}", "", "", "", ""])
rows.append(["H1: below the status tier (%)", *[ci(s, "below the status tier (linked)") for s, _ in SC],
             f"{N.loc[('primary', 'area', 'below the status tier (linked)'), 'fmi']:.2f}"])
rows.append(["H2: status tier, floor-area share ÷ building shareᶜ", *[rd(s, "h2b", lambda x: f"{x:.2f}") for s, _ in SC],
             f"{RD.loc[('primary', 'h2b'), 'fmi']:.2f}"])
rows.append(["H3: at the thermostat-and-meter profile (%)", *[ci(s, "at the minimal code-required profile") for s, _ in SC],
             f"{N.loc[('primary', 'area', 'at the minimal code-required profile'), 'fmi']:.2f}"])
rows.append(["H3: Shapley R² share, construction period − building typeᵈ",
             *[sg(rd(s, "h4", lambda x: f"{x:.3f}")) for s, _ in SC], f"{RD.loc[('primary', 'h4'), 'fmi']:.2f}"])
write("table_headline", "AI readiness and hypothesis estimates for the 2018 U.S. commercial stock.",
      ["Quantity", "Strict (low)ᵇ", "Primary", "Enriched (high)", "Missing\ninformationᵃ"], rows,
      r"L{3.6cm}L{2.0cm}L{2.1cm}L{2.0cm}L{2.0cm}",
      f"Floor-area weighted unless stated. Brackets: 95% intervals that combine sampling and imputation uncertainty ({MI}). "
      "—: not computed for the scenario.\n"
      "ᵃ The fraction of missing information of the primary estimate: the share of its variance due to the imputed code "
      "edition, recording and linkage.\n"
      "ᵇ Acts at the linked level only. Its code-required and declared levels are the primary's, up to the imputation draw.\n"
      "ᶜ H2 holds where the ratio exceeds 1.\n"
      "ᵈ At or below zero, construction period explains no more of the variance than building type, which supports H3 without confirming it.")

# ---------------------------------------------------------------- zero profile
LBL = {"<= 5,000": "≤ 5,000 ft²", "5,001-24,999": "5,001–24,999 ft²", ">= 25,000": "≥ 25,000 ft²",
       "smart thermostat": "Smart thermostat", "AMI-recorded meter": "Advanced meter"}
Z = pd.read_csv(D / "zero_profile.csv")
rows = []
for cut, title in (("SIZE", "Size class"), ("PBA", "Building type (largest five)"), ("YRCONC", "Construction period"), ("CENDIV", "Census division"),
                   ("attribute", "Equipmentᵃ")):
    z = Z[(Z.cut == cut) & (Z.population == "zero linked level")].set_index("statistic")
    s = Z[(Z.cut == cut) & (Z.population == "whole stock")].set_index("statistic")
    keys = list(z.index)
    if cut == "PBA":
        keys = list(z.estimate.sort_values(ascending=False).index[:5])
    for i, k in enumerate(keys):
        lab = LBL.get(str(k), str(k)).replace(" to ", "–") if cut == "YRCONC" else LBL.get(str(k), str(k))   # en-dash, as in the text
        rows.append([title if i == 0 else "", lab, f"{100 * z.loc[k, 'estimate']:.1f} [{100 * z.loc[k, 'lo95']:.1f}, {100 * z.loc[k, 'hi95']:.1f}]",
                     f"{100 * s.loc[k, 'estimate']:.1f}"])
write("table_zero_profile", "Where the floor area with no linked data lies, by size, type, period and division, against the "
      "whole stock.", ["", "Group", "Floor area with no linked data (%)", "Whole stock (%)"], rows,
      r"L{2.4cm}L{3.8cm}L{3.0cm}L{1.8cm}",
      f"Primary result. Each column gives the distribution of its floor area over the groups. Brackets: 95% intervals ({MI}).\n"
      "ᵃ Not part of the distribution: the share of the column's floor area with a BAS, a smart thermostat or an advanced meter.")

print("wrote table_headline and table_zero_profile")
