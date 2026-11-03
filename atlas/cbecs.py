"""Load the CBECS 2018 public microdata (pinned), keep the variables the atlas uses, label them.

Response codes come from the pinned EIA codebook (``atlas.codebook``), never from memory. Every
"Missing" in a variable the atlas uses is a skip, not a non-response (imputed values are filled by
EIA and flagged in the Z* columns); ``UNIVERSE`` states who was asked each question, and
``check_universes`` verifies those statements against the file, so they are tested facts.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from atlas import codebook, config

# variable -> (who was asked, the rule as a predicate on the frame that must equal "not missing")
UNIVERSE: dict[str, tuple[str, object]] = {
    "RENOV": ("buildings constructed before 2013 (YRCONC < 9)", lambda d: d.YRCONC < 9),
    "RENHVC": ("buildings with RENOV = 1", lambda d: d.RENOV == 1),
    "RENLGT": ("buildings with RENOV = 1", lambda d: d.RENOV == 1),
    "RENELC": ("buildings with RENOV = 1", lambda d: d.RENOV == 1),
    "EMCS": ("buildings with any heating or cooling (HEATP > 0 or COOLP present)",
             lambda d: (d.HEATP.fillna(0) > 0) | d.COOLP.notna()),
    "EMCSHT": ("EMCS = 1 and the building is heated (HEATP present)",
               lambda d: (d.EMCS == 1) & d.HEATP.notna()),
    "EMCSCL": ("EMCS = 1 and the building is cooled (COOLP present)",
               lambda d: (d.EMCS == 1) & d.COOLP.notna()),
    "EMCSLT": ("EMCS = 1 (one EMCS=1 record unasked)", None),
    "SMRTTHRM": ("buildings WITHOUT a BAS (EMCS = 2) only", lambda d: d.EMCS == 2),
    "SCHED": ("lit buildings with electricity", None),
    "OCSN": ("lit buildings with electricity", None),
    "DIM": ("lit buildings with electricity", None),
    "DCV": ("buildings with air distribution (397 records unasked, incl. the 243 unconditioned)", None),
    "VAV": ("as DCV", None),
    "DOAS": ("as DCV", None),
    "DAYHARV": ("lit buildings with electricity (same universe as OCSN)", lambda d: d.OCSN.notna()),
    "OWNOCC": ("not applicable for 85 records (mostly vacant or unowned-occupancy cases)", None),
}

YESNO = ("EMCS", "EMCSHT", "EMCSCL", "EMCSLT", "SMRTTHRM", "SCHED", "OCSN", "DIM",
         "DCV", "VAV", "DOAS", "DAYHARV", "RENOV", "RENHVC", "RENLGT", "RENELC",
         # used by the exploratory enriched Table E readings
         "ECN", "CHILLR", "BOILER", "CHWT", "STHW", "DRLGHT", "ACVRF", "ENRGYPLN")
CATEGORICAL = ("REGION", "CENDIV", "PBA", "PBAPLUS", "PUBCLIM", "SQFTC", "YRCONC", "ELUSED",
               "OWNTYPE", "OWNOCC") + YESNO
NUMERIC = ("SQFT", "FINALWT", "HEATP", "COOLP", "NFLOOR", "MONUSE")
FLAGS = tuple("Z" + v for v in ("YRCONC", "RENOV", "SQFT", "EMCS", "EMCSHT", "EMCSCL", "EMCSLT",
                                "SMRTTHRM", "SCHED", "OCSN", "DIM", "DCV", "VAV", "DOAS",
                                "DAYHARV", "OWNTYPE", "OWNOCC"))
REPLICATES = tuple(f"FINALWT{i}" for i in range(1, 152))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pins() -> None:
    for path in (config.CBECS_CSV, config.CODEBOOK_XLSX):
        got = sha256(path)
        if got != config.SHA256[path.name]:
            raise ValueError(f"{path.name}: sha256 {got} != pinned {config.SHA256[path.name]}")


@lru_cache(maxsize=1)
def load(replicates: bool = False) -> pd.DataFrame:
    """The atlas columns of the pinned file, one row per building, indexed by PUBID."""
    verify_pins()
    cols = ["PUBID", *CATEGORICAL, *NUMERIC, *FLAGS, *(REPLICATES if replicates else ())]
    df = pd.read_csv(config.CBECS_CSV, usecols=cols, low_memory=False).set_index("PUBID")
    if len(df) != config.N_RECORDS:
        raise ValueError(f"{len(df)} records, expected {config.N_RECORDS}")
    return df


def label(var: str, code) -> str:
    """Codebook label of a response code; NaN -> 'not asked'."""
    if pd.isna(code):
        return "not asked"
    return codebook.variables()[var].codes.get(str(int(code)), f"?{code}")


def check_universes(df: pd.DataFrame) -> dict[str, bool]:
    """Each stated universe holds exactly: asked <=> the rule (rules given as None are descriptive)."""
    return {v: bool((df[v].notna() == rule(df)).all())
            for v, (_, rule) in UNIVERSE.items() if rule is not None}


def family_of_pbaplus(df: pd.DataFrame) -> dict[int, int]:
    """PBAPLUS -> PBA. The atlas's 'PBAPLUS family' is EIA's own PBA grouping; this checks nesting."""
    pairs = df.groupby("PBAPLUS")["PBA"].unique()
    bad = {k: list(v) for k, v in pairs.items() if len(v) != 1}
    if bad:
        raise ValueError(f"PBAPLUS not nested in PBA: {bad}")
    return {int(k): int(v[0]) for k, v in pairs.items()}
