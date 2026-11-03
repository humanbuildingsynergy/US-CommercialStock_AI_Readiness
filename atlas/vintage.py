"""Table V: construction-year category (+ renovation) -> ASHRAE 90.1 edition in force.

Inputs are hand-authored, cited assumption tables (every assumption a row):

  tables/V_editions.csv   the edition ladder: edition, year published, DOE determination, source
  tables/V_states.csv     per state, the year each 90.1-equivalent edition came into force
                          (PNNL-19887 Table 7.1, PNNL-31437 Table A.1, PNNL-25611 for MS; DOE
                          status snapshots as lower bounds for CA/OR/WA, which PNNL excludes)
  tables/V_renovation.csv which renovation subtypes re-date which building systems (the codebook
                          gives no renovation year, only "since 2000 or since constructed")

Rule (stated in the table output): a state's edition at year y is the newest edition whose
year_in_force <= y (pre-1989 if none); a division's edition at y is the lower median over its
states (unweighted). A construction bin or renovation window gets the division edition at its
midpoint year (central), +-1 rung on the ladder (the adoption-lag interval), with the upper end
clamped to editions published by the window end. The public file has no state, so the division
value is a prior over its states, never any building's actual code.

    python -m atlas.vintage      -> table_V.{md,csv} (an earlier design's outputs, not shipped)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from atlas import config

YRCONC_YEARS = {  # codebook YRCONC bounds; the open lower bin gets a nominal start (no code before)
    2: (1900, 1945), 3: (1946, 1959), 4: (1960, 1969), 5: (1970, 1979), 6: (1980, 1989),
    7: (1990, 1999), 8: (2000, 2012), 9: (2013, 2018),
}
SURVEY_YEAR = 2018
RENOV_FROM = 2000  # codebook RENOV: "since the year 2000 or since it was constructed"
SYSTEMS = ("hvac", "lighting", "electrical")


def _rows(name: str) -> list[dict[str, str]]:
    with open(config.TABLES / name, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if not r[next(iter(r))].startswith("#")]


@lru_cache(maxsize=1)
def ladder() -> tuple[str, ...]:
    """Editions in order, oldest first (index = rung)."""
    return tuple(r["edition"] for r in _rows("V_editions.csv"))


@lru_cache(maxsize=1)
def published() -> dict[str, int]:
    return {r["edition"]: int(r["published"]) for r in _rows("V_editions.csv")}


# Table V variants, each run as a sensitivity (atlas.sensitivities). The primary table has all three False.
VARIANT = {"drop_implicit": False, "state_weighted": False, "doe_dates": False}


def configure(drop_implicit: bool = False, state_weighted: bool = False, doe_dates: bool = False) -> None:
    """Switch the Table V variant and clear every cache that depends on it. doe_dates (sensitivity S13): the 17
    state-edition pairs where DOE's dates disagree with PNNL-31437's reported-actual years take DOE's year
    (tables/V_states_doe_dates.csv)."""
    VARIANT.update(drop_implicit=drop_implicit, state_weighted=state_weighted, doe_dates=doe_dates)
    state_rows.cache_clear()
    edition_at.cache_clear()


@lru_cache(maxsize=1)
def state_rows() -> dict[str, tuple[int, list[tuple[int, str]]]]:
    """state -> (cendiv, [(year_in_force, edition)])."""
    out: dict[str, tuple[int, list]] = {}
    doe = {(r["state"], r["edition"]): int(r["doe_year"]) for r in _rows("V_states_doe_dates.csv")} if VARIANT["doe_dates"] else {}
    for r in _rows("V_states.csv"):
        row = out.setdefault(r["state"], (int(r["cendiv"]), []))
        if VARIANT["drop_implicit"] and r["kind"] == "implicit":
            continue
        row[1].append((doe.get((r["state"], r["edition"]), int(r["year_in_force"])), r["edition"]))
    return out


@lru_cache(maxsize=1)
def state_weights() -> dict[str, float]:
    """State weights: 2018 commercial-sector energy consumption per state (EIA SEDS TECCB)."""
    return {r["state"]: float(r["teccb_2018_billion_btu"]) for r in _rows("V_state_weights.csv")}


def state_edition_at(state: str, year: int) -> str:
    L = ladder()
    have = [L.index(e) for y, e in state_rows()[state][1] if y <= year]
    return L[max(have, default=0)]


def division_states(cendiv: int) -> list[str]:
    return sorted(s for s, (d, _) in state_rows().items() if d == cendiv)


@lru_cache(maxsize=None)
def edition_at(cendiv: int, year: int) -> str:
    """Lower median of the division's states' rungs at *year*: unweighted over states (primary),
    or weighted by state commercial energy use (the first rung whose cumulative weight
    reaches half the division's total)."""
    L = ladder()
    pairs = sorted((L.index(state_edition_at(s, year)), s) for s in division_states(cendiv))
    if not VARIANT["state_weighted"]:
        return L[pairs[(len(pairs) - 1) // 2][0]]
    w = state_weights()
    total, acc = sum(w[s] for _, s in pairs), 0.0
    for rung, s in pairs:
        acc += w[s]
        if acc >= total / 2:
            return L[rung]
    return L[pairs[-1][0]]


def divisions() -> list[int]:
    return sorted({d for d, _ in state_rows().values()})


@dataclass(frozen=True)
class Interval:
    low: str
    central: str
    high: str

    def rungs(self) -> tuple[int, int, int]:
        L = ladder()
        return L.index(self.low), L.index(self.central), L.index(self.high)


def interval(cendiv: int, start: int, end: int) -> Interval:
    """Central = division edition at the window midpoint; +-1 rung, clamped to what was published
    by the window end (and to the ladder)."""
    L = ladder()
    newest = max(i for i, e in enumerate(L) if published()[e] <= end)
    c = min(L.index(edition_at(cendiv, (start + end) // 2)), newest)
    return Interval(L[max(c - 1, 0)], L[c], L[min(c + 1, newest)])


def renovation_window(yrconc: int) -> tuple[int, int]:
    start = max(RENOV_FROM, YRCONC_YEARS[yrconc][0])
    return start, SURVEY_YEAR


@lru_cache(maxsize=1)
def renovation_rules() -> dict[str, tuple[str, ...]]:
    """renovation subtype variable -> building systems it re-dates."""
    return {r["variable"]: tuple(r["systems"].split("+"))
            for r in _rows("V_renovation.csv") if r["systems"]}


def table_v() -> pd.DataFrame:
    """Table V: one row per (YRCONC, division) for construction, and per (YRCONC, division) for a
    renovation in that bin."""
    rows = []
    for yc, (a, b) in YRCONC_YEARS.items():
        for cd in divisions():
            iv = interval(cd, a, b)
            rows.append({"basis": "construction", "yrconc": yc, "years": f"{a}-{b}" if yc > 2 else f"<{b + 1}",
                         "cendiv": cd, "midpoint": (a + b) // 2,
                         "low": iv.low, "central": iv.central, "high": iv.high})
            if yc < 9:  # RENOV not asked of 2013-2018 buildings
                ra, rb = renovation_window(yc)
                iv = interval(cd, ra, rb)
                rows.append({"basis": "renovation", "yrconc": yc, "years": f"{ra}-{rb}",
                             "cendiv": cd, "midpoint": (ra + rb) // 2,
                             "low": iv.low, "central": iv.central, "high": iv.high})
    return pd.DataFrame(rows)


def assign(df: pd.DataFrame) -> pd.DataFrame:
    """Per record and building system: the edition rung (low, central, high) in force.

    Construction sets every system; a renovation subtype re-dates the systems Table V-R names,
    and only upward (a renovation is later than construction). ``new_*`` is the construction
    rung alone, for clauses that bind new construction only."""
    tv = table_v().set_index(["basis", "yrconc", "cendiv"])
    L = ladder()
    out = {}
    for pubid, r in df.iterrows():
        con = tv.loc[("construction", int(r.YRCONC), int(r.CENDIV))]
        base = tuple(L.index(con[k]) for k in ("low", "central", "high"))
        sysr = {s: base for s in SYSTEMS}
        if r.RENOV == 1:
            ren = tv.loc[("renovation", int(r.YRCONC), int(r.CENDIV))]
            rr = tuple(L.index(ren[k]) for k in ("low", "central", "high"))
            for var, systems in renovation_rules().items():
                if r[var] == 1:
                    for s in systems:
                        sysr[s] = tuple(max(x, y) for x, y in zip(sysr[s], rr))
        out[pubid] = {f"{s}_{k}": sysr[s][i] for s in SYSTEMS
                      for i, k in enumerate(("low", "central", "high"))}
        out[pubid].update({f"new_{k}": base[i] for i, k in enumerate(("low", "central", "high"))})
    return pd.DataFrame.from_dict(out, orient="index")


DIVISION_NAMES = {1: "New England", 2: "Middle Atlantic", 3: "East North Central",
                  4: "West North Central", 5: "South Atlantic", 6: "East South Central",
                  7: "West South Central", 8: "Mountain", 9: "Pacific"}
