"""The question canon and the answerable-readiness ceiling C(I), copied from the readiness framework.

COPIED, not imported: the requirement table ``_V2``, the stream vocabulary and ``ceiling``
are verbatim from the canon module of the readiness framework (Jung 2026; code and data at
github.com/humanbuildingsynergy/building-ai-readiness, release v1.0.0, DOI 10.5281/zenodo.23197610).
Cite the framework for them; never redefine them here.

  v2  the framework's 76-level canon (31 domain x intent cells, up to three answer levels each), the one this study
      uses; ``atlas/tests/test_floors.py`` checks the copy against the framework's release file when it is fetched.

A canon is a requirement table: for every (domain, intent) cell, the stream classes needed for
each answer-completeness level; C(I) = sum(achieved levels) / sum(caps).
"""
from __future__ import annotations

from dataclasses import dataclass

DOMAINS = ("thermal", "energy", "airquality", "equipment", "occupancy", "lighting")
INTENTS = ("status", "summary", "comparison", "verification", "diagnosis", "explanation")

Cell = tuple[str, str]
Rungs = tuple[frozenset[str], ...]


def _rows(*rows: tuple[str, tuple[str, ...], tuple[str, ...]]) -> dict[Cell, Rungs]:
    out: dict[Cell, Rungs] = {}
    for domain, intents, levels in rows:
        rungs = tuple(frozenset(lv.split("+")) for lv in levels)
        for intent in intents:
            out[(domain, intent)] = rungs
    return out


# ── v2 (the framework's canon module): the 76-cell canon ─────────────────────────────────────────
_V2 = _rows(
    ("thermal", ("status", "summary"), ("T", "T+Sp")),
    ("thermal", ("comparison",), ("T", "T", "T+Sp+ActC+ActP")),
    ("thermal", ("verification",), ("T", "T+Sp", "T+Sp+ActC+ActP")),
    ("thermal", ("diagnosis", "explanation"), ("T", "T+Sp", "T+Sp+SpC+ActC+ActP+SAT")),
    ("energy", ("status",), ("Whole",)),
    ("energy", ("summary",), ("Whole", "Whole+Sub")),
    ("energy", ("comparison", "diagnosis", "explanation"), ("Whole", "Whole", "Whole+Sub2+OAT")),
    ("airquality", ("status", "summary"), ("CO2", "CO2+Occ")),
    ("airquality", ("comparison",), ("CO2", "CO2", "CO2+Occ+OAF")),
    ("airquality", ("verification", "diagnosis", "explanation"), ("CO2", "CO2+Occ", "CO2+Occ+OAF")),
    ("equipment", ("status", "summary"), ("Pwr", "Pwr")),
    ("equipment", ("comparison", "verification", "diagnosis", "explanation"),
     ("Pwr", "Pwr", "Pwr+Run+Load")),
    ("occupancy", ("status", "summary", "comparison", "diagnosis"), ("Occ", "Occ")),
    ("lighting", ("status", "diagnosis"), ("Lgt", "Lgt+Occ")),
    ("lighting", ("summary", "comparison"), ("Lgt", "Lgt")),
)

# ── the HVAC-credit scoring, this study's sensitivity reading (not the framework's) ──────────────
# The equipment questions also accept the points a building automation system reads. A cell's level under this scoring
# is the higher of its primary level and its level on these alternative rungs, so equipment power keeps every level it
# reaches under the primary and no building scores lower. All other cells are the primary's.
_V2H_ALT = _rows(
    ("equipment", ("status", "summary"), ("Run", "Run+Load")),
    ("equipment", ("comparison", "verification", "diagnosis", "explanation"), ("Run", "Run+Load", "Load+Run+SAT")),
)

# v2 stream-class vocabulary (the readiness framework's legend).
STREAMS: dict[str, str] = {
    "T": "zone air temperature",
    "Sp": "zone temperature setpoint in effect (heating or cooling)",
    "SpC": "scheduled (occupied/unoccupied) setpoint held apart from the setpoint in effect",
    "ActC": "terminal actuation command (damper or valve command or setpoint, heating or cooling command)",
    "ActP": "terminal actuation response (damper or valve position, terminal airflow)",
    "SAT": "supply air temperature of the air system serving the zone",
    "OAT": "outside air temperature",
    "Occ": "occupancy (presence or count)",
    "CO2": "zone CO2 concentration",
    "OAF": "outdoor-air flow",
    "Whole": "whole-building electricity meter",
    "Sub": "at least one end-use submeter (HVAC, lighting, plug loads, thermal energy)",
    "Sub2": "two end-use groups metered (the third is the residual of the whole)",
    "Pwr": "equipment input power (electricity or fuel)",
    "Run": "equipment run command or status",
    "Load": "equipment load (air or water flow, delivered heating or cooling)",
    "Lgt": "lighting power, lighting status, or illuminance",
}

# The framework's catalog-id -> stream-class map (its canon module's _V2_IDS).
V2_IDS = {
    "Zone_Air_Temperature_Sensor": "T",
    "Zone_Air_Cooling_Temperature_Setpoint": "Sp", "Zone_Air_Heating_Temperature_Setpoint": "Sp",
    "Occupancy_Sensor": "Occ", "CO2_Sensor": "CO2",
    "Whole_Building_Electrical_Energy": "Whole", "Lighting_Electrical_Power": "Lgt",
    "Electrical_Power_Sensor": "Pwr",
}
V2_END_USE = {"Pwr": "hvac", "Lgt": "lighting"}


def end_use_streams(groups: set[str]) -> set[str]:
    """The readiness framework's derived v2 classes Sub / Sub2 from the set of metered end-use groups."""
    return ({"Sub"} if groups else set()) | ({"Sub2"} if len(groups) >= 2 else set())


def catalog_streams(catalog_ids, canon: str = "v2") -> set[str]:
    ids = set(catalog_ids)
    have = {V2_IDS[i] for i in ids if i in V2_IDS}
    return have | end_use_streams({V2_END_USE[s] for s in have if s in V2_END_USE})


@dataclass(frozen=True)
class Canon:
    name: str
    rungs: dict[Cell, Rungs]

    @property
    def total(self) -> int:
        return sum(len(r) for r in self.rungs.values())

    def achieved_level(self, domain: str, intent: str, have: set[str]) -> int:
        level = 0
        for r, required in enumerate(self.rungs.get((domain, intent), ()), start=1):
            if required <= have:
                level = r
        return level


CANONS: dict[str, Canon] = {"v2": Canon("v2", _V2), "v2h_alt": Canon("v2h_alt", _V2H_ALT)}


def level(domain: str, intent: str, have: set[str], scoring: str = "primary") -> int:
    """Achieved level of one cell under the primary scoring or the HVAC-credit scoring."""
    lv = CANONS["v2"].achieved_level(domain, intent, have)
    if scoring == "hvac" and (domain, intent) in _V2H_ALT:
        lv = max(lv, CANONS["v2h_alt"].achieved_level(domain, intent, have))
    return lv


@dataclass(frozen=True)
class Ceiling:
    canon: str
    streams: tuple[str, ...]
    by_domain: dict[str, tuple[int, int]]
    achieved: int
    total: int

    @property
    def score(self) -> float:
        return self.achieved / self.total


# ── how much the question set's weighting matters (this study's sensitivity readings) ────────────
# Every cell keeps its primary level; only the weights change. "domain_equal": mean over the six domains of (levels
# reached in the domain ÷ levels available in it); "intent_equal": the same over the six intents; "domain_equal_tiers":
# domain_equal, with the tier coverages also weighting each domain's question type equally within an intent.
SCORINGS = ("primary", "hvac", "domain_equal", "intent_equal", "domain_equal_tiers")


def readiness(have, scoring: str = "primary") -> float:
    """AI readiness C(I) of the stream-class set *have* under a scoring: levels reached ÷ 76 (primary, and the
    HVAC-credit scoring), or the domain- or intent-equal means."""
    if scoring in ("primary", "hvac"):
        return ceiling(have, "v2h" if scoring == "hvac" else "v2").score
    c, have = CANONS["v2"], set(have)
    part = 1 if scoring == "intent_equal" else 0          # cell = (domain, intent)
    keys = INTENTS if part else DOMAINS
    got = {k: 0 for k in keys}
    cap = {k: 0 for k in keys}
    for cell, rungs in c.rungs.items():
        got[cell[part]] += c.achieved_level(*cell, have)
        cap[cell[part]] += len(rungs)
    return sum(got[k] / cap[k] for k in keys) / len(keys)


def ceiling(have, canon: str = "v2") -> Ceiling:
    """Answerable-readiness ceiling of the stream-class set *have* (the readiness framework's definition); canon
    "v2h" is the HVAC-credit scoring on the v2 cells."""
    c, have = CANONS["v2" if canon == "v2h" else canon], set(have)
    per_cell = {cell: (level(*cell, have, "hvac") if canon == "v2h" else c.achieved_level(*cell, have)) for cell in c.rungs}
    by_domain = {d: (sum(v for (dd, _), v in per_cell.items() if dd == d),
                     sum(len(r) for (dd, _), r in c.rungs.items() if dd == d)) for d in DOMAINS}
    return Ceiling(canon, tuple(sorted(have)), by_domain, sum(per_cell.values()), c.total)
