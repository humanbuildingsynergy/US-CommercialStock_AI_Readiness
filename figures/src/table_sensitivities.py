"""Results table of the one-at-a-time sensitivities (main text, Results 4.4), from the multiple imputation in
results/estimates/sensitivities.csv. Point estimates only; Appendix Table E.1 (appendix_tables.py) keeps the
intervals and the other statistics. Writes figures/tables/table_sensitivities.tex/.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import tables as T1   # the shared writer only

ROOT = Path(__file__).resolve().parents[2]
S = pd.read_csv(ROOT / "results" / "estimates" / "sensitivities.csv")
M = int(S["M"].max())

# Rows in the order of the premises in Table 2 (tab:inputs) that each tests; S7 (reweighted reference set) could not be
# run and is left out.
ROWS = [
    ("primary", "Primary result"),
    ("S19", "Question set, domains weighted equally"),
    ("S20", "Question set, intents weighted equally"),
    ("S11", "Imputed control answers dropped (contrast)"),
    ("S1", "States weighted equally within divisions"),
    ("S2", "No rows assumed for states without a code"),
    ("S13", "DOE adoption years (17 state–edition pairs)"),
    ("S15", "Editions applied 1 year after adoption"),
    ("S16", "Editions applied 2 years after adoption"),
    ("S3", "Occupancy sensing from 90.1-2004, 2010"),
    ("S17", "No monitoring credit in non-90.1 states"),
    ("S5", "Controls table as first written (no CO2, daylighting or VAV rows, BAS lighting as occupancy)"),
    ("S6", "Beta(0.5, 0.5) prior for rare streams"),
    ("S21", "Graphs that link none missing (all 9)"),
    ("S22", "The same, only the 7 without time series"),
    ("S8", "Monthly bills credited"),
    ("S9", "Smart thermostats not credited"),
    ("S10", "Six controls credited without a BAS"),
    ("S12", "Advanced-meter shares of 2025"),
    ("S18", "Meter only with daily digital access"),
    ("S14", "Interval meter if ≥ 25,000 ft²"),
]


def val(sid: str, stat: str, fmt) -> str:
    r = S[(S.id == sid) & (S.statistic == stat)]
    if r.empty or pd.isna(r.estimate.iloc[0]):
        return "—"
    return fmt(float(r.estimate.iloc[0]))


pct = lambda x: f"{100 * x:.1f}"
three = lambda x: f"{x:.3f}".replace("-", "−")
two = lambda x: f"{x:.2f}"

rows = [[lab,
         val(sid, "mean linked level", three),
         val(sid, "below the status tier (linked)", pct),
         val(sid, "linked level zero", pct),
         val(sid, "H2b concentration ratio", two),
         val(sid, "H4 vintage - type", three)] for sid, lab in ROWS]

T1.write("table_sensitivities", "The primary result and the hypothesis estimates under each sensitivity, varied one at a time.",
         ["Varied", "Mean\nlevel", "Below\nstatus", "No\ndata", "H2\nratioᵃ", "H3\ndiff.ᵇ"], rows,
         r"L{6.7cm}L{1.02cm}L{1.02cm}L{1.02cm}L{1.02cm}L{1.15cm}",
         f"Point estimates under the full multiple imputation (M = {M}, Rubin's rules). Table~\\ref{{tab:app-E1}} gives their 95% intervals "
         "and the tier shares. Mean level: the mean linked level. Below status and No data: the shares of floor area (%) below the "
         "status tier and with no linked data. —: not computed for the contrast.\n"
         "ᵃ H2 holds where the concentration ratio exceeds 1. "
         "ᵇ Construction period minus building type, which supports H3 at or below zero.")
