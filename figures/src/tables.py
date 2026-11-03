"""Generated manuscript tables, from committed files (LaTeX for the paper, Markdown for review).

The shared table writer; the tables themselves are written by tables_v2.py and table_sensitivities.py.
Writes figures/tables/<id>.tex and <id>.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def n_rows(name: str) -> int:
    with open(ROOT / "tables" / name, encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def n_states() -> int:
    with open(ROOT / "tables" / "V_states.csv", encoding="utf-8") as f:
        return len({r[next(iter(r))] for r in csv.DictReader(f)})


def write(fid: str, caption: str, header: list[str], rows: list[list[str]], colspec: str, note: str = "") -> None:
    flat = lambda c: c.replace("\n", " ")   # "\n" marks a deliberate line break in a cell (LaTeX \newline)
    md = [f"**{caption}**", "", "| " + " | ".join(flat(h) for h in header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(flat(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", f"*{note}*"]
    (OUT / f"{fid}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def tex(s: str) -> str:
        return (s.replace("%", r"\%").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("±", r"$\pm$")
                .replace("→", r"$\rightarrow$").replace("×", r"$\times$").replace("÷", r"$\div$").replace("−", r"$-$")
                .replace("R²", r"$R^2$").replace("§", r"\S").replace("ft²", r"ft\textsuperscript{2}").replace("–", "--")
                .replace("CO2", r"CO$_2$").replace("\n", r"\newline ")
                .replace("ᵃ", r"\textsuperscript{a}").replace("ᵇ", r"\textsuperscript{b}").replace("ᶜ", r"\textsuperscript{c}").replace("ᵈ", r"\textsuperscript{d}"))   # note markers
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize\linespread{1}\selectfont", rf"\caption{{{tex(caption)}}}",
             rf"\label{{tab:{fid}}}", r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [rf"\par\smallskip\parbox{{\textwidth}}{{\footnotesize {tex(note)}}}"]   # left-aligned, one line per marked note
    lines += [r"\end{table}"]
    (OUT / f"{fid}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote figures/tables/{fid}.tex/.md")

