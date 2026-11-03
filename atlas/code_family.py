"""The code family (ASHRAE 90.1, IECC or a state's own code) of every row of tables/V_states.csv.

V_states records the 90.1-equivalent edition in force, not which code a state adopted. The family is read from the DOE
Building Energy Codes Program's own naming of each state's commercial code (tables/V_code_text.csv: one row per state
and record, from the archived DOE state pages of 2019-2021 and the quarterly state-level workbooks of 2020-2026, each
with its source):

  - "90.1" when the code named first is ASHRAE 90.1 (e.g. "ASHRAE Standard 90.1-2013");
  - "IECC" when it is an IECC edition (e.g. "2015 IECC with Amendments", "2015 IECC and 90.1-2013": the IECC lets a
    building comply through 90.1, but the state's code is the IECC);
  - "state" for a state's own code (California Title 24, Florida, Indiana, Oregon, Washington);
  - "none" where DOE lists no statewide code; "implicit" for the implicit-practice rows (no adopted code).

A row takes the family of the earliest DOE record dated in or after its year in force; rows in force before the
earliest record (mid-2019) take that record. The first record is the code in force at that date, so a state that
changed family between adopting an edition and 2019 is assigned its 2019 family (Michigan's "2015 IECC and 90.1-2013" is
classed IECC with doubt). Used only by the sensitivity that withholds the 90.1-2013 monitoring credit outside 90.1
states (S17).

    python -m atlas.code_family     -> the code_family columns of tables/V_states.csv, from tables/V_code_text.csv
"""
from __future__ import annotations

import csv
import re

import pandas as pd

from atlas import config


def family(text: str) -> str:
    t = str(text).strip().lower()
    if not t or t in ("nan", "parse_fail") or t.startswith("<"):   # "<90.1-2010" is an equivalence legend, not a code name
        return ""
    if "none statewide" in t or "home rule" in t or "no statewide" in t:
        return "none"
    if ("state specific" in t or "state energy code" in t or "title 24" in t or "bees" in t
            or "building energy efficiency standards" in t):
        return "state"
    m_ashrae, m_iecc = re.search(r"ashrae|^90\.1|^asrae", t), re.search(r"iecc", t)
    if m_ashrae and (not m_iecc or m_ashrae.start() < m_iecc.start()):
        return "90.1"
    if m_iecc:
        return "IECC"
    return ""


def assign() -> pd.DataFrame:
    T = pd.read_csv(config.TABLES / "V_code_text.csv")
    T = T[T.family.notna() & (T.family != "")]
    p = config.TABLES / "V_states.csv"
    V = pd.read_csv(p, dtype={"edition": str})
    fam, src = [], []
    for r in V.itertuples():
        if r.kind == "implicit":
            fam.append("implicit")
            src.append("implicit-practice row (no adopted code)")
            continue
        t = T[T.state == r.state]
        later = t[t.date >= f"{int(r.year_in_force)}-01-01"]
        pick = (later if not later.empty else t).iloc[0] if not t.empty else None
        if pick is None:
            fam.append("unknown")
            src.append("no DOE record")
        else:
            fam.append(pick.family)
            src.append(f"{pick.code_text} ({pick.date}; {pick.source})")
    V["code_family"], V["code_family_source"] = fam, src
    V.to_csv(p, index=False, quoting=csv.QUOTE_MINIMAL)
    return V


if __name__ == "__main__":
    V = assign()
    print(V.code_family.value_counts())
