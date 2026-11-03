"""The appendix table bodies, generated from the result tables as figures/tables/appendix/app-<letter><n>.tex, one per
table of the paper's appendix.

Each file holds only the body (size commands, the tabular, and the table note); the caption and label stay in the
appendix file. A table taller than a page is a longtable, which cannot sit in a float: its file carries the caption and
label (CAPTIONS below) and replaces the whole table environment in the appendix (needs \\usepackage{longtable}).
Style as the manuscript tables (booktabs, footnotesize, tabcolsep 4pt, the L{} column type of main.tex); manuscript
vocabulary (code-required, declared and linked level; primary result; strict and enriched scenarios; knowledge graphs).

    python figures/src/appendix_tables.py
"""
from __future__ import annotations

import csv
import functools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atlas import canon  # noqa: E402

OUT = ROOT / "figures" / "tables" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)
T = ROOT / "tables"
W6 = ROOT / "results" / "estimates"
M = 200
CAPTIONS = {   # the paper's captions of the four longtables (LaTeX)
    "A2": "The code-clause table: each clause, the streams it requires, when it binds and its sources.",
    "B1": "The reported-controls table: each survey question, the streams it declares, how many reference-set knowledge "
          "graphs declare them, and the reasoning.",
    "D1": "The data source and parameter draw of each imputed layer.",
    "E1": r"AI readiness and tier shares (Panel A) and the hypothesis estimates (Panel B) under each sensitivity, with 95\% "
          "intervals."}


def tex(s) -> str:
    s = str(s)
    s = (s.replace("\\", "/").replace("{", r"\{").replace("}", r"\}").replace("&", r"\&\allowbreak{}").replace("%", r"\%")
         .replace("_", r"\_").replace("#", r"\#").replace(">=", "≥").replace("<=", "≤").replace(">", "$>$").replace("<", "$<$")
         .replace("|", r"$\vert$").replace("+", r"+\allowbreak{}").replace("/", r"/\allowbreak{}")
         .replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$").replace("→", r"$\rightarrow$")
         .replace("±", r"$\pm$").replace("–", "--").replace("−", "$-$").replace("δ", r"$\delta$").replace("²", r"$^2$")
         .replace("§", r"\S{}").replace("CO2", r"CO$_2$").replace("½", "1/2").replace("∝", r"$\propto$").replace("÷", r"$\div$")
         .replace("ᵃ", r"\textsuperscript{a}").replace("ᵇ", r"\textsuperscript{b}").replace("ᶜ", r"\textsuperscript{c}"))   # note markers
    s = re.sub(r"(^|[\s(])'", r"\1`", s)
    s = re.sub(r"\[\[fig-([a-z]+)\]\]", r"Figure~\\ref{fig:fig_\1}", s)   # [[fig-comparison]] -> Figure~\ref{fig:fig_comparison}
    return re.sub(r"\[\[([A-G]\d)\]\]", r"Table~\\ref{tab:app-\1}", s)   # [[D1]] -> Table~\ref{tab:app-D1}


def scrub(s) -> str:
    """The paper's vocabulary in cells: no file paths; the reference set and its knowledge graphs named as such."""
    s = str(s)
    s = re.sub(r"\s*\(provisional\)", "", s)
    s = re.sub(r"linkage draw\b", "linkage imputation", s)
    s = re.sub(r"(?:tables/)?\b\w+\.csv(?:, )?", "", s).replace("()", "").replace("( ", "(").replace(" ,", ",")
    s = s.replace("year_in_force", "year in force")
    s = re.sub(r"(?<!knowledge )\b(census )?graphs\b", lambda m: (m.group(1) or "") + "knowledge graphs", s)
    s = re.sub(r"\bcensus(?= (?:knowledge|frequenc|parameters|linkage|rates?|shares?|odds|test|bound|graphs?))", "reference-set", s)
    s = re.sub(r"\bthe census\b(?! division)", "the reference set", s)
    s = re.sub(r"\bAMI\b", "advanced-meter", s)
    return s


PLAIN = (   # Table B.1's reasoning in plain words (no decision labels, row codes or class codes)
    (r"\.?\s*Grade: [^.]*$", ""), (r"\s*\(last class missing[^)]*\);?\s*also reported alone", ""), (r";\s*Table C row \w+ sources [\w/]+", ""),
    (r";\s*the discount carries this", ""), (r"\s*\(EMCS = 2\)", ""), (r"the canon's Lgt class includes", "the lighting stream includes"),
    (r"\(Lgt includes lighting status\)", "(the lighting stream includes lighting status)"),
    (r"the canon's end-use submeter class names thermal energy", "the end-use submeter stream includes thermal energy"),
    (r"\bSAT presumes", "Supply-air temperature presumes"), (r"\bDCV\b", "demand-controlled ventilation"),
    (r"a DOAS", "a dedicated outdoor-air system"), (r"a VRF central", "a variable-refrigerant-flow (VRF) central"),
    (r"\bWITHOUT\b", "without"), (r"declared, not linked: ", ""), (r"no stream assumed: ", "no stream assumed; "),
)


def plain(s: str) -> str:
    s = scrub(s)
    for a, b in PLAIN:
        s = re.sub(a, b, s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(";,")
    return s[:1].upper() + s[1:] + ("" if s.endswith(".") else ".")


def caption_of(label: str) -> str:
    return CAPTIONS[label]


def table(label: str, header: list[str], rows: list, colspec: str, note: str = "", raw: bool = False, long: bool = False,
          repeat_head: bool = True) -> None:
    note = r"\newline ".join(tex(part) for part in note.split("\n"))   # one line per marked note
    head = " & ".join(rf"\textbf{{{tex(h)}}}" for h in header) + r" \\"
    size = [r"\footnotesize\linespread{1}\selectfont", r"\setlength{\tabcolsep}{4pt}"]
    body = []
    for r in rows:
        body.append(r"\midrule" if r == "MIDRULE" else " & ".join(c if raw or isinstance(r, Raw) else tex(scrub(c)) for c in r) + r" \\")
    src = f"% Generated by figures/src/appendix_tables.py from committed results; do not edit by hand."
    if long:
        n = len(header)
        L = [src, "% longtable: replaces the whole table environment of tab:app-" + label + " in 08_appendix.tex.",
             r"\begingroup", *size, rf"\begin{{longtable}}{{{colspec}}}",
             rf"\caption{{{caption_of(label)}}}\label{{tab:app-{label}}} \\", r"\toprule", head, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{n}}}{{l}}{{\textit{{Table \thetable{{}} (continued)}}}} \\", r"\toprule", *([head, r"\midrule"] if repeat_head else []), r"\endhead",
             r"\midrule", rf"\multicolumn{{{n}}}{{r}}{{\textit{{continued on the next page}}}} \\", r"\endfoot", r"\bottomrule",
             r"\endlastfoot", *body, r"\end{longtable}"]
        if note:
            L.append(rf"\vspace{{-0.5\baselineskip}}\noindent\parbox{{\textwidth}}{{\scriptsize {note}}}\par")
        L.append(r"\endgroup")
    else:
        L = [src, *size, rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
        if note:
            L.append(rf"\par\smallskip\parbox{{\textwidth}}{{\scriptsize {note}}}")
    (OUT / f"app-{label}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote app-{label}.tex" + (" (longtable)" if long else ""))


def stk(est: str, ci: str) -> str:
    return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{tex(est)}\\{{\scriptsize {tex(ci)}}}\end{{tabular}}"


def ci3(r) -> tuple[str, str]:
    if min(r.estimate, r.lo95) < 0:   # signed: a comma, not a dash, between the ends
        return f"{r.estimate:+.3f}".replace("-", "−"), f"[{r.lo95:+.3f}, {r.hi95:+.3f}]".replace("-", "−")
    return f"{r.estimate:.3f}", f"[{r.lo95:.3f}, {r.hi95:.3f}]"


def cip(r) -> tuple[str, str]:
    return f"{100 * r.estimate:.1f}", f"[{100 * r.lo95:.1f}, {100 * r.hi95:.1f}]"


def short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


class Raw(list):
    """A row already in LaTeX (panel headings): written as is, in any table."""


def panel(title: str, heads: list[str]) -> list:
    return ["MIDRULE", Raw([rf"\textbf{{{tex(title)}}}", *[rf"\textbf{{{tex(h)}}}" for h in heads]]), "MIDRULE"]


DIV = {1: "New England", 2: "Middle Atlantic", 3: "East North Central", 4: "West North Central", 5: "South Atlantic",
       6: "East South Central", 7: "West South Central", 8: "Mountain", 9: "Pacific"}

# ── A1 editions by division ─────────────────────────────────────────────────────────────────────
vs = pd.read_csv(T / "V_states.csv")
sw = pd.read_csv(T / "V_state_weights.csv").set_index("state")
ra = []
for d, name in DIV.items():
    s = vs[vs.cendiv == d]
    states = sorted(s.state.unique())
    cells = []
    for e in ("1989", "2004", "2010", "2013"):
        later = {"1989": ("1989", "1999/2001", "2001", "2004", "2007", "2010", "2013", "2016"), "2004": ("2004", "2007", "2010", "2013", "2016"),
                 "2010": ("2010", "2013", "2016"), "2013": ("2013", "2016")}[e]
        hit = s[s.edition.astype(str).isin(later) & (s.kind != "implicit")]
        first = hit.groupby("state").year_in_force.min()
        cells.append("—" if first.empty else f"{int(first.min())} ({int((first <= 2018).sum())})")
    w = sw.loc[states, "teccb_2018_billion_btu"]
    ra.append([name, str(len(states)), *cells, f"{w.idxmax()} ({100 * w.max() / w.sum():.0f})"])
table("A1", ["Census division", "States", "90.1-1989", "90.1-2004", "90.1-2010", "90.1-2013", "Largest state (%)ᵃ"],
      ra, r"L{2.7cm}L{1.45cm}L{1.45cm}L{1.45cm}L{1.45cm}L{1.45cm}L{1.6cm}",
      "First year any state of the division had the edition, a later one or its IECC equivalent in force, with the number of "
      "states that had it by 2018 in parentheses, including states placed at that edition or later by a DOE status "
      "snapshot.\nᵃ The state with the most commercial energy use in 2018 "
      "(EIA State Energy Data System) and its share of the division, the weight used to draw a building's state.")

# ── A2 clauses ──────────────────────────────────────────────────────────────────────────────────
TITLES = {r["code"]: r["title"] for r in csv.DictReader(open(T / "C_sources.csv", encoding="utf-8"))}


def source(s: str) -> str:
    """The clause table's source codes as readable citations (the codes resolve in tables/C_sources.csv)."""
    out = []
    for piece in re.split(r";\s*|,\s*(?=S\d)", s):
        code = re.match(r"(S\d+)", piece)
        add = re.search(r"\((\d\d) ([a-z]+)", piece)
        if add:
            c = f"90.1-20{add.group(1)} addendum {add.group(2)}"
        elif code and code.group(1) == "S1":
            c = "PNNL-17722"
        elif code and code.group(1) in ("S2", "S3", "S4", "S5", "S6", "S7"):
            c = "DOE determination, " + TITLES[code.group(1)].split()[0]
        elif code:
            c = TITLES[code.group(1)].split()[0]
        else:
            c = piece
        if c not in out:
            out.append(c)
    return "; ".join(out)


USE = {"core": "all results", "occ_late": "all results", "occ_alt": "sensitivity only", "meter_clause": "all results",
       "not_used": "not used",
       "occ_early": "sensitivity only",
       "sensitivity": "not used (condition not in the survey)"}
COND = {"conditioned": "conditioned space", "electric": "electricity used", "lit": "lighting installed",
        "lit&sqft>5000": "lighting installed, over 5,000 ft²", "electric&sqft>=25000": "electricity used, at least 25,000 ft²",
        "electric&sqft>=25000&not-hotel-motel-restaurant": "electricity used, at least 25,000 ft², not a hotel, motel or restaurant",
        "sqft>=25000&district": "at least 25,000 ft², district chilled water, steam or hot water used",
        "air system >= 10 hp fan power or plant >= 300,000 Btu/h serving > 3 zones (unobservable)": "fan power or plant capacity (not in the survey)",
        "electric chilled-water plant above the climate-zone capacity (unobservable)": "chilled-water plant capacity (not in the survey)",
        "hotel or motel with more than 50 guest rooms (not applied)": "hotel or motel, more than 50 rooms (not applied)",
        "dense-spaces (unobservable)": "densely occupied spaces (not in the survey)",
        "economizer-required (unobservable)": "economizer required (not in the survey)",
        "replaced-equipment (unobservable)": "equipment replaced (not in the survey)"}
# Demand-controlled ventilation never mandates a CO2 sensor; occupancy-based control also complies.
# The row enters no estimate, so the clause table keeps its CO2 code and only the printed table says so.
SHOWN = {"C11": "CO2 or Occ", "C15": "Sub (district energy only)", "C16": "none", "C17": "none", "C18": "none"}
rows = [[r["id"], r["clause"], SHOWN.get(r["id"], r["streams"]), r["rung_from"], COND.get(r["condition"], r["condition"]), USE.get(r["reading"], r["reading"]),
         source(r["source"])]
        for r in csv.DictReader(open(T / "C_clauses.csv", encoding="utf-8"))]
table("A2", ["Row", "Clause", "Streams", "From", "Condition", "Use", "Source"], rows,
      r"lL{2.85cm}L{1.5cm}L{1.0cm}L{1.8cm}L{1.6cm}L{2.2cm}",
      "Stream codes as in [[C1]]. From: the first 90.1 edition with the clause (pre-1989: universal practice). Condition: when the "
      "clause binds, in survey terms. Use: all results = the primary result and both scenarios; not used = enters no estimate. Rows C07, C09 and C10, the "
      "readiness framework's own reading of 90.1-2022, are not used and are not shown. The full requirement text and the "
      "quoted passages are deposited with the data and code.", long=True)

# ── B1 reported controls ────────────────────────────────────────────────────────────────────────
pres = pd.read_csv(T / "E_controls_presence.csv").set_index("id")
ROWUSE = {"wo": "all three", "extended": "all three", "enriched": "enriched", "enriched_spc": "enriched",
          "wo_bas_occ": "sensitivity only", "enriched_weak": "not used", "none": "no stream"}
QUESTION_B1 = {"E19": "Chiller or boiler plant, with a BAS that controls heating or cooling?"}   # the CSV holds a shorthand
ROWUSE_ID = {"E19": "optimistic only", "E20": "sensitivity only"}   # E19 enters the optimistic scenario (atlas/declared.py); E20 only S10
rows = []
for r in csv.DictReader(open(T / "E_controls.csv", encoding="utf-8")):
    k = pres.loc[r["id"], "presence_k"]
    why = plain(r["assumption"])
    grade = re.search(r"Grade: ([a-z-]+) tier", r["assumption"] + " " + r["linked_note"])   # graded before any effect was computed
    use = ROWUSE_ID.get(r["id"], ROWUSE.get(r["reading"], r["reading"])) + (f" ({grade.group(1).replace('-', ' to ')})" if grade else "")
    rows.append([r["id"], QUESTION_B1.get(r["id"], r["cbecs_question"]), r["streams"] if r["streams"].strip("-– ") else "none",
                 "—" if pd.isna(k) or k == "" else f"{int(float(k))} of 45", use, why])
table("B1", ["Row", "Survey question", "Streams", "Declaring", "Use", "Reasoning"], rows,
      r"lL{2.95cm}L{1.5cm}L{1.75cm}L{1.55cm}L{3.4cm}",
      "Declaring: how many of the reference set's 45 knowledge graphs declare every stream of the row. Use: all three = the primary result "
      "and the strict and enriched scenarios; the strict scenario removes, from the automation-system routes, every stream that fewer "
      "than a tenth of the knowledge graphs declare. In parentheses: the row's grade, how strongly the equipment implies the stream, set "
      "before any effect was computed. Reasoning: the row's assumption; the full wording, with its sources, is "
      "deposited with the data and code.", long=True)

# ── B2 reference-set linkage ───────────────────────────────────────────────────────────────────────────
lc = pd.read_csv(T / "R_linkage_classes.csv")
lg = pd.read_csv(T / "R_linkage_graphs.csv")
pos = lg[lg.in_R1]
part = pos[pos.type == "partial"].ratio
rows = [[f"Link all their declared streams", str(int((pos.type == "all").sum())), "", "", "", ""],
        [f"Link none", str(int((pos.type == "none").sum())), "", "", "", ""],
        [f"Link part (linked share {part.min():.2f}–{part.max():.2f})", str(int((pos.type == "partial").sum())), "", "", "", ""]]
rows += panel("Panel B: stream", ["Declaredᵃ", "Linked", "Ratio", "Partial: declared / linked", "Rate usedᵇ"])
RC = pd.read_csv(ROOT / "results" / "scoring" / "reference_set_counts.csv").set_index("stream")
SHORT = {"T": "zone temperature", "Sp": "setpoint in effect", "SpC": "scheduled setpoint", "ActC": "terminal actuation command",
         "ActP": "terminal actuation response", "SAT": "supply-air temperature", "OAT": "outside-air temperature", "Occ": "occupancy",
         "CO2": "zone CO2", "OAF": "outdoor-air flow", "Whole": "whole-building meter", "Sub": "end-use submeter",
         "Sub2": "two end-use groups metered", "Pwr": "equipment power", "Run": "run status", "Load": "equipment load",
         "Lgt": "lighting"}   # short names, not truncations; the definitions are Table C.1's legend
for r in lc.itertuples():
    if r.declared_R1 == 0:
        continue
    rc = RC.loc[r[1]]
    rows.append([f"{r[1]}: {SHORT[r[1]]}", str(r.declared_R1), str(r.linked_R1), f"{r.lgd_R1:.2f}",
                 f"{int(rc['declared, 6 partial'])} / {int(rc['linked, 6 partial'])}", f"{rc['partial rate used (point)']:.2f}"])
table("B2", ["Panel A: knowledge graphs", "Numberᵃ", "", "", "", ""], rows, r"L{4.6cm}rrrL{1.5cm}r",
      "A stream is linked when it carries a time-series reference in the published "
      "graph. Stream definitions: [[C1]]. Ratio: linked ÷ declared.\n"
      f"ᵃ Counts, in both panels, over the {len(pos)} reference-set knowledge graphs whose declared streams answer at least one question (the 45 "
      f"less {45 - len(pos)} whose declared streams answer none).\n"
      "ᵇ The stream's linked share among the 6 partly linked graphs, the rate at which a "
      "partly linked building links it (Beta(1, 1) posterior mean where fewer than five of them declare the stream; each "
      "imputation draws the rate).")

# ── C1 question set ─────────────────────────────────────────────────────────────────────────────
rows, seen = [], {}
for (dom, intent), rungs in canon._V2.items():
    seen.setdefault((dom, tuple("+".join(sorted(x)) for x in rungs)), []).append(intent)
for (dom, lv), intents in seen.items():
    rows.append([dom, ", ".join(intents), *[lv[i] if i < len(lv) else "—" for i in range(3)]])
legend = "; ".join(f"{k} {v}" for k, v in canon.STREAMS.items())
table("C1", ["Domain", "Intents", "Level 1", "Level 2", "Level 3"], rows, r"L{1.6cm}L{3.0cm}L{1.8cm}L{2.2cm}L{3.4cm}",
      f"31 question types (domain × intent) with 76 answer levels; intents with the same levels share a row. —: the question type has fewer levels. Stream codes: {legend}.")

# ── D1 imputation models ────────────────────────────────────────────────────────────────────────
PLAIN_D1 = (   # Table C.1 in plain words: no survey variable names or imputation jargon
    (r"^state ∝ ", "state drawn in proportion to its "),
    (r"construction year uniform within the YRCONC period", "construction year drawn uniformly within the survey's construction period"),
    (r"edition = newest with year[ _]in[ _]force\s*(<=|≤)\s*year\s*;?", "edition: the newest in force in that state by that year;"),
    (r"renovated systems re-dated over \[max\(2000, period start\), 2018\]",
     "a renovated system takes the edition of a year drawn from 2000, or its period's start if later, to 2018"),
    (r"linkage type all / none / partial and per-class linked-given-declared rates",
     "the shares of knowledge graphs that link all, none or part of their streams, and each class's linked share among those "
     "that declare it,"),
    (r"proper imputation: each imputation refits the type and class rates on a bootstrap resample",
     "each imputation re-estimates these shares on a resample, with replacement,"),
    (r"draws rare-class rates \(< 5 declarations\)", "draws the share of a class that fewer than five declare"),
    (r"type and class linkage then drawn per building", "each building's linkage is then drawn"),
    (r"EMCS with heating or cooling control \(BAS route\) or SMRTTHRM = 1 \(smart thermostat\)",
     "a BAS that controls heating or cooling, or a reported smart thermostat"),
    (r"\s*\(C08\)", ""),
    (r"reported donors in the same building type × size class × construction period cell \((?:≥|>=) 10 donors; coarser cells otherwise\); "
     r"EMCS, EMCSHT, EMCSCL, EMCSLT, SMRTTHRM, OCSN, DCV, VAV, DOAS, DAYHARV; the BAS block moves together when EMCS was imputed",
     "buildings of the same type, size class and construction period that reported the answer (at least 10, or a coarser group "
     "otherwise), for the BAS and its heating, cooling and lighting control, smart thermostat, occupancy sensors, "
     "demand-controlled ventilation, VAV, dedicated outdoor-air system and daylight harvesting; when the BAS answer was "
     "imputed, its control answers are redrawn with it"),
    (r"redrawn per imputation \(hot deck\)", "redrawn from a donor in every imputation"))
plain_d1 = lambda t: functools.reduce(lambda a, pr: re.sub(pr[0], pr[1], a), PLAIN_D1, t)
LAYER_D1 = {"§8.4.3 recording": "monitoring-clause recording"}
rows = [[LAYER_D1.get(r["imputation model"], r["imputation model"]), plain_d1(scrub(r["data source"])),
         plain_d1(scrub(r["parameter-draw method"]))]
        for r in csv.DictReader(open(T / "G_imputation_models.csv", encoding="utf-8"))]
table("D1", ["Layer", "Data source", "Parameter draw"], rows, r"L{2.2cm}L{6.2cm}L{4.4cm}", "—: none.", long=True)

# ── D2 model checks ─────────────────────────────────────────────────────────────────────────────
mc = pd.read_csv(W6 / "model_checks.csv")
NAME = {"all": "knowledge graphs linking all their data", "none": "knowledge graphs linking none", "partial": "knowledge graphs linking part",
        "partial ratio min": "smallest linked share, partial", "partial ratio max": "largest linked share, partial"}
rows = []
for r in mc.itertuples():
    lab = NAME.get(r.statistic, r.statistic.replace("linked count, ", "knowledge graphs linking stream "))
    f = (lambda x: f"{x:.2f}") if "share" in lab else (lambda x: f"{x:.0f}")
    rows.append([lab, f(r.observed), f"{f(mc.loc[r.Index, 'simulated 2.5%'])}–{f(mc.loc[r.Index, 'simulated 97.5%'])}",
                 f"{mc.loc[r.Index, 'two-sided p']:.2f}"])
assert mc.inside.all()   # the note says every statistic is inside its range
table("D2", ["Statistic", "Observed", "Simulated 95% range", "Two-sided p"], rows, r"L{6.0cm}rL{2.6cm}r",
      "2,000 simulated reference sets from the fitted linkage model, each with its own parameter draw. Every observed value lies "
      "inside its simulated range. Two-sided p: twice the smaller of the shares of simulated sets at or above and at or below "
      "the observed value, capped at 1.")

# ── E1 sensitivities ────────────────────────────────────────────────────────────────────────────
S = pd.read_csv(W6 / "sensitivities.csv") if (W6 / "sensitivities.csv").exists() else pd.DataFrame(columns=["id"])
PANELS = ((("mean linked level", False), ("status tier (linked)", True), ("summary tier (linked)", True), ("diagnosis tier (linked)", True)),
          (("below the status tier (linked)", True), ("H2b concentration ratio", False), ("diagnosis tier (declared)", True),
           ("H4 vintage - type", False)))
HEADS = (["Mean linked level", "Status tier (%)", "Summary tier (%)", "Diagnosis tier (%)"],
         ["H1: below the status tier (%)", "H2: concentration ratioᵃ", "Diagnosis tier, declared (%)ᵇ", "H3: period − typeᶜ"])
# Sensitivities are named in words, never by label.
SENS_SHORT = {"S1": "Equal state weights", "S2": "No assumed adoption rows", "S3": "Occupancy from 2004 and 2010",
              "S24": "Occupancy from 2016 lighting alterations",
              "S5": "Controls table as first written", "S6": "Jeffreys prior for rare streams",
              "S7": "Reweighted reference set", "S8": "Monthly bills credited", "S9": "No smart thermostats",
              "S10": "Six controls without a BAS", "S11": "Imputed answers dropped", "S12": "2025 advanced meters",
              "S13": "DOE adoption dates", "S14": "Interval meters, large buildings", "S15": "Adoption lag, 1 year",
              "S16": "Adoption lag, 2 years", "S17": "Monitoring credit only under 90.1", "S18": "Daily digital meter access",
              "S19": "Domains weighted equally", "S20": "Intents weighted equally",
              "S21": "Link-none graphs missing (all 9)", "S22": "Link-none graphs missing (7)"}
SENS_DESC = {"S3": "Occupancy sensing credited from 90.1-2004 and 2010",
             "S24": "Occupancy sensing credited for lighting altered under 90.1-2016",
             "S5": "Controls table as first written: no rows for demand-controlled ventilation, daylight harvesting or VAV, and BAS lighting control read as occupancy sensing",
             "S19": "Each domain of the question set weighted equally", "S20": "Each intent of the question set weighted equally",
             "S6": "Rare-stream prior Beta(0.5, 0.5)",
             "S21": "Knowledge graphs that link none treated as missing (all 9)",
             "S22": "Knowledge graphs that link none treated as missing (only the 7 with no time-series reference)",
             "S13": "DOE's adoption dates for the 17 state-edition pairs where DOE and the federal laboratory analysis disagree",
             "S14": "Interval metering credited for every building of at least 25,000 ft²",
             "S15": "Edition in force 1 year before construction or renovation",
             "S16": "Edition in force 2 years before construction or renovation",
             "S17": "No 90.1-2013 monitoring credit where the state's code was IECC or a state code",
             "S18": "Whole-building meter recorded with the state's share of commercial customers with daily digital access"}
SENS_DROPPED = {"S23"}   # a scenario dimension (d on BAS streams), reported in panel C of the scenario grid below
# Rows in the order of the premises in Table 2 (tab:inputs) that each tests, as in the main-text table_sensitivities.
SENS_ORDER = ["primary", "S19", "S20", "S11", "S1", "S2", "S13", "S15", "S16", "S3", "S24", "S17", "S5", "S6", "S21", "S22", "S8", "S9", "S10", "S12", "S18", "S14"]
rows = []
for p, (stats, heads) in enumerate(zip(PANELS, HEADS)):
    if p:
        rows += panel("Panel B", heads)
    for sid in [*SENS_ORDER, *(i for i in dict.fromkeys(S.id) if i not in SENS_ORDER and i not in SENS_DROPPED)]:
        g = S[S.id == sid].set_index("statistic")
        lab = "Primary result" if sid == "primary" else (SENS_DESC.get(sid, g.varied.iloc[0]) if p == 0 else SENS_SHORT.get(sid, sid))
        if "not run" in g.index:   # e.g. the reweighted reference set: not possible (no reference-set graph has a public type, size and period)
            continue
        cell = lambda k, pct: stk(*(cip(g.loc[k]) if pct else ci3(g.loc[k]))) if k in g.index and not pd.isna(g.loc[k, "estimate"]) else "—"
        rows.append([tex(lab), *[cell(k, pct) for k, pct in stats]])
table("E1", ["Panel A", *HEADS[0]], rows, r"L{3.0cm}L{2.1cm}L{2.15cm}L{2.15cm}L{2.45cm}",
      f"Primary result and each sensitivity under the full multiple imputation ({M} imputations, Rubin's rules, 95% intervals in "
      "brackets). Every interval follows Rubin's rules within its own "
      "sensitivity; shares on the logit scale, H2 on the log scale. Dropping imputed answers does not recalibrate the weights. The 2025 advanced-meter shares change only that input; everything else stays at 2018. Daily digital access is commercial customers with daily access ÷ commercial customers (EIA-861, 2018).\n"
      "ᵃ Status-tier share of floor area ÷ share of buildings.\nᵇ The diagnosis tier at the declared level.\n"
      "ᶜ Construction period minus building type in the Shapley R² decomposition. At or below zero, construction period explains no more of the variance than building type, which supports H3 without confirming it.",
      raw=True, long=True, repeat_head=False)   # two panels: a repeated Panel A heading would sit over Panel B

# ── E2 scenario grid ────────────────────────────────────────────────────────────────────────────
# description applies to non-BAS routes only (the design), one draw per stream
G = pd.read_csv(ROOT / "results" / "scenarios" / "scenario_grid_stream.csv")
G = G[G.scenario == "primary"] if "scenario" in G else G
rows = []
for c in (0.5, 0.75, 1.0):
    for d in (0.5, 0.75, 1.0):
        g = G[(G.c.round(2) == c) & (G.d.round(2) == d) & (G.da == 0) & (G.db == 0)].set_index("statistic")
        if g.empty:
            continue
        m, b = g.loc["mean linked level"], g.loc["below the status tier (linked)"]
        rows.append([f"{c:.2f}", f"{d:.2f}", f"{m.estimate:.3f} [{m.lo95:.3f}, {m.hi95:.3f}]", f"{100 * b.estimate:.1f} [{100 * b.lo95:.1f}, {100 * b.hi95:.1f}]"])
# the full range (0, 1], one dimension at a time with the other at 1 (d = 0 not run: nothing described, level 0)
GF = G
bound = False   # set when an interval is given as a bound
rows += panel("Panel B, over (0, 1]: compliance c", ["Description d", "Mean linked level", "Below the status tier (%)"])
for axis in ("c", "d"):
    for v in (0.05, 0.10, 0.25, 0.50, 0.75):
        g = GF[np.isclose(GF[axis], v) & np.isclose(GF["d" if axis == "c" else "c"], 1.0)].set_index("statistic")
        m, b = g.loc["mean linked level"], g.loc["below the status tier (linked)"]
        bound = bound or b.estimate > 0.9999
        bl = (f"{100 * b.estimate:.1f} [≥ 99.99]" if b.estimate > 0.9999 else
              f"{100 * b.estimate:.1f} [{100 * b.lo95:.1f}, {100 * b.hi95:.1f}]")
        rows.append([f"{v:.2f}" if axis == "c" else "1.00", f"{v:.2f}" if axis == "d" else "1.00",
                     f"{m.estimate:.3f} [{m.lo95:.3f}, {m.hi95:.3f}]", bl])
# S23: d applied to every linked stream (BAS routes included), c = 1
GB = pd.read_csv(ROOT / "results" / "scenarios" / "scenario_grid_bas_description.csv")
rows += panel("Panel C, d on BAS streams too: compliance c", ["Description d", "Mean linked level", "Below the status tier (%)"])
for v in (0.05, 0.10, 0.25, 0.50, 0.75):
    g = GB[np.isclose(GB.d, v) & np.isclose(GB.c, 1.0)].set_index("statistic")
    m, b = g.loc["mean linked level"], g.loc["below the status tier (linked)"]
    bound = bound or b.estimate > 0.9999
    bl = (f"{100 * b.estimate:.1f} [≥ 99.99]" if b.estimate > 0.9999 else
          f"{100 * b.estimate:.1f} [{100 * b.lo95:.1f}, {100 * b.hi95:.1f}]")
    rows.append(["1.00", f"{v:.2f}", f"{m.estimate:.3f} [{m.lo95:.3f}, {m.hi95:.3f}]", bl])
table("E2", ["Panel A: compliance c", "Description d", "Mean linked level", "Below the status tier (%)"], rows, r"L{2.6cm}lL{3.6cm}L{3.6cm}",
      f"Primary result, {M} imputations, Rubin's rules, 95% intervals in brackets; c = d = 1 is the primary result. Description "
      "d applies to data recorded outside the BAS (smart thermostat, advanced meter, monitoring clause); BAS points already carry "
      "it through their reference-set linkage. Panel A: the "
      "grid the main analysis reports. Panel B: each dimension alone down to 0.05 with the other at 1; d = 0 would leave nothing "
      "described. Panel C: d applied to BAS points as well, with c at 1." + (" Where almost all floor area is below the status tier the interval is given as a bound." if bound else ""))
