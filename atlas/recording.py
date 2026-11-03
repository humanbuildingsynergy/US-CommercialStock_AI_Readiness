"""The recording layer: the meter tables, the route table and the per-building route masks.

A stream counts only if it is installed, recorded and described:
  installed  code clauses (with compliance) and reported controls (with presence), as before;
  recorded   its readings are stored as a time series through a route CBECS lets us see;
  described  a stored series can be matched to its equipment or space (held at 1, swept).

Routes (``tables/L_recording.csv``, one row per source row x stream classes):
  BAS trend logs        classes a building automation system reads; the linkage draw (all / none /
                        partial, census class rates) is now read as this recording route
  smart thermostat      zone temperature and setpoint where SMRTTHRM = 1 (asked only without a BAS)
  utility interval data the whole-building meter, recorded with the 2018 commercial advanced-meter share of
                        the building's state (EIA-861; ``tables/L_ami_states.csv``), the same state as the
                        edition draw
  code-required logging the 90.1-2013 §8.4.3 streams, total electricity included, where the clause applies
                        (record every 15 min, keep 36 months; Table C row C08), subject to compliance
  none                  conventional thermostats; local occupancy or lighting sensors without a BAS
                        that controls lighting; the flagged rows whose route CBECS leaves unclear (by default)

    python -m atlas.recording tables     -> tables/L_recording.csv, tables/L_ami_states.csv
    python -m atlas.recording ami 2025   -> tables/L_ami_states_2025.csv (a later EIA-861 year)
    python -m atlas.recording daily 2018 -> tables/L_daily_access_states_2018.csv (daily digital access)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from atlas import config, declared, vintage

AMI_ZIP = config.SOURCES / "eia861" / "f8612018.zip"
AMI_XLSX = config.SOURCES / "eia861" / "Advanced_Meters_2018.xlsx"
AMI_URL = "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612018.zip"
EPA_1010_URL = "https://www.eia.gov/electricity/annual/html/epa_10_10.html"
D_SWEEP = (1.0, 0.75, 0.5)
FLAGGED = ("E07", "E12", "E13", "E17", "E18", "E20")   # route unclear in CBECS; the provisional rule is final

# ── the advanced-meter table ─────────────────────────────────────────────────────────────────────
def _ami_columns(rows: list[dict]) -> tuple[str, str, str]:
    """State, commercial AMI and commercial total-meter columns, found by header (the layout moves between years)."""
    group, cols = "", {}
    key = lambda c: sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(c)))
    for c in sorted(set(rows[0]) | set(rows[1]), key=key):
        group = rows[0].get(c) or group
        cols[(group.strip(), (rows[1].get(c) or "").strip())] = c
    return (cols[("Utility Characteristics", "State")], cols[("Number AMI- Advanced Metering Infrastructure", "Commercial")],
            cols[("Total Numbers of Meters", "Commercial")])


def ami_xlsx(year: int = 2018):
    return config.SOURCES / "eia861" / f"Advanced_Meters_{year}.xlsx"


def ami_table(year: int = 2018):
    return config.TABLES / ("L_ami_states.csv" if year == 2018 else f"L_ami_states_{year}.csv")


def ami_rows(year: int = 2018) -> list[dict]:
    from atlas.codebook import read_xlsx
    allrows = read_xlsx(ami_xlsx(year))
    cs, cm, ct = _ami_columns(allrows)
    rows = [r for r in allrows[2:] if r.get("A") == str(year)]
    num = lambda s: float(s) if s not in (".", "", None) else 0.0
    ami, tot, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for r in rows:
        s = r[cs]
        ami[s] += num(r[cm])        # AMI, commercial
        tot[s] += num(r[ct])        # total meters, commercial
        n[s] += 1
    return [{"state": s, "ami_commercial": int(ami[s]), "meters_commercial": int(tot[s]),
             "ami_share": round(ami[s] / tot[s], 6) if tot[s] else "", "utility_state_rows": n[s],
             "source": f"EIA-861 {year}, Advanced_Meters_{year}.xlsx: AMI (col {cm}) / total meters (col {ct}), commercial, summed "
                       "over the utility-state rows of the state (a utility serving several states reports one row per "
                       "state, so each part counts in its own state)"} for s in sorted(ami)]


def daily_table(year: int = 2018):
    return config.TABLES / f"L_daily_access_states_{year}.csv"


def daily_rows(year: int = 2018) -> list[dict]:
    """Commercial customers with daily digital access ("able to access daily energy usage through a web
    portal or other electronic means"; Advanced_Meters file) over commercial customers served by a distribution utility
    (Sales_Ult_Cust file: bundled service, parts A and D, plus delivery-only service, part C; energy-only service, part
    B, is the same customers counted again by their retail supplier, so it is left out). Customers over customers, not
    customers over meters: a customer can have several meters, and access is granted per customer account."""
    from atlas.codebook import read_xlsx
    am = read_xlsx(ami_xlsx(year))
    group, cols = "", {}
    key = lambda c: sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(c)))
    for c in sorted(set(am[0]) | set(am[1]), key=key):
        group = am[0].get(c) or group
        cols[(group.strip(), (am[1].get(c) or "").strip())] = c
    cs, cd = cols[("Utility Characteristics", "State")], cols[("Customers with Daily Digital Access", "Commercial")]
    num = lambda s: float(s) if s not in (".", "", None) else 0.0
    daily = defaultdict(float)
    for r in am[2:]:
        if r.get("A") == str(year):
            daily[r[cs]] += num(r[cd])
    su = read_xlsx(config.SOURCES / "eia861" / f"Sales_Ult_Cust_{year}.xlsx")
    cust = defaultdict(float)
    for r in su[3:]:
        if r.get("A") == str(year) and r.get("D") in ("A", "C", "D"):
            cust[r["G"]] += num(r["O"])     # commercial customers
    shares = ami_share(year)
    return [{"state": s, "daily_access_commercial": int(daily[s]), "customers_commercial": int(cust[s]),
             "daily_access_share": round(min(daily[s] / cust[s], 1.0), 6) if cust[s] else "",
             "ami_share_of_meters": shares.get(s, ""),
             "source": f"EIA-861 {year}: Advanced_Meters_{year}.xlsx 'Customers with Daily Digital Access', Commercial (col {cd}) "
                       f"over Sales_Ult_Cust_{year}.xlsx commercial customers (col O), parts A, C and D"} for s in sorted(cust)]


def build_daily_year(year: int) -> None:
    rows = daily_rows(year)
    with open(daily_table(year), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    D = sum(r["daily_access_commercial"] for r in rows)
    C = sum(r["customers_commercial"] for r in rows)
    print(f"{year}: commercial customers with daily digital access {D:,} / {C:,} = {D / C:.4f}; states {len(rows)}")


@lru_cache(maxsize=4)
def daily_share(year: int = 2018) -> dict[str, float]:
    with open(daily_table(year), encoding="utf-8", newline="") as f:
        return {r["state"]: float(r["daily_access_share"]) for r in csv.DictReader(f) if r["daily_access_share"] != ""}


@lru_cache(maxsize=4)
def ami_share(year: int = 2018) -> dict[str, float]:
    with open(ami_table(year), encoding="utf-8", newline="") as f:
        return {r["state"]: float(r["ami_share"]) for r in csv.DictReader(f)}


# ── the recording table ──────────────────────────────────────────────────────────────────────────
BAS_HVAC = "EMCS==1 & (EMCSHT==1 | EMCSCL==1)"
BAS_LIGHT = "EMCS==1 & EMCSLT==1"
ROUTES = [
    # source, classes, route, condition, rule, flag, source/evidence
    ("C01", "T+Sp", "BAS trend logs", BAS_HVAC, "reference-set linkage imputation (type, then class rates)", "",
     "a thermostat's readings are stored only by a BAS that controls heating or cooling"),
    ("C01", "T+Sp", "smart thermostat", "SMRTTHRM==1", "recorded (deterministic)", "",
     "SMRTTHRM is asked only of buildings without a BAS (EMCS = 2); 3,387 inapplicable = 3,144 with a BAS + 243 unconditioned"),
    ("C01", "T+Sp", "none", "otherwise", "not recorded", "",
     "a conventional thermostat stores nothing. Includes 16 BAS buildings whose BAS controls neither heating nor cooling (never asked SMRTTHRM)"),
    ("C02", "Whole", "code-required logging", "C08 applies and is complied with", "recorded", "",
     "90.1-2013 §8.4.3 monitors total electrical energy, recorded every 15 min, kept 36 months (Table C C08)"),
    ("C02", "Whole", "utility interval data", "otherwise", "recorded with probability = 2018 commercial AMI share of the drawn state",
     "", "only interval data counts; tables/L_ami_states.csv (EIA-861 2018). Access to the data is covered by the description sweep"),
    ("C05, C06", "Occ", "BAS trend logs", BAS_LIGHT, "reference-set linkage imputation", "",
     "occupancy sensors record only through a networked route; a BAS that controls lighting"),
    ("C05, C06", "Occ", "none", "otherwise", "not recorded", "", "wall-box sensors act locally and store nothing"),
    ("C08", "Lgt+Pwr+Sub+Sub2", "code-required logging", "clause applies and is complied with", "recorded", "",
     "90.1-2013 §8.4.3: record every 15 min, keep 36 months (Table C C08)"),
    ("E01", "T+Sp+SAT+Run", "BAS trend logs", "row condition (BAS controls heating or cooling)", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E02", "Occ", "BAS trend logs", "row condition (BAS controls lighting)", "reference-set linkage imputation", "", "sensitivity reading only (S5)"),
    ("E03", "Occ", "BAS trend logs", BAS_LIGHT, "reference-set linkage imputation", "", "occupancy and lighting sensors record only through a networked route"),
    ("E03", "Occ", "none", "otherwise", "not recorded", "", "wall-box sensors act locally"),
    ("E04", "T+Sp", "smart thermostat", "row condition (SMRTTHRM = 1)", "recorded (deterministic)", "", "a smart thermostat stores its readings"),
    ("E07", "CO2", "BAS trend logs (provisional)", BAS_HVAC, "reference-set linkage imputation", "provisional rule, kept as final",
     "a DCV CO2 sensor may feed a stand-alone unit controller that stores nothing; provisional: recorded only through a BAS"),
    ("E07", "CO2", "none (provisional)", "otherwise", "not recorded", "provisional rule, kept as final", "as above"),
    ("E08", "Lgt", "BAS trend logs", BAS_LIGHT, "reference-set linkage imputation", "", "a daylight photosensor records only through a networked route"),
    ("E08", "Lgt", "none", "otherwise", "not recorded", "", "occupancy and lighting sensors record only through a networked route"),
    ("E09", "ActC+ActP", "BAS trend logs", "row condition (BAS + VAV)", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E10", "Lgt", "BAS trend logs", "row condition (BAS controls lighting)", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E12", "OAT", "BAS trend logs (provisional)", BAS_HVAC, "reference-set linkage imputation", "provisional rule, kept as final",
     "an economizer's outdoor sensor may sit in a packaged unit's own controller; provisional: recorded only through a BAS"),
    ("E12", "OAT", "none (provisional)", "otherwise", "not recorded", "provisional rule, kept as final", "as above"),
    ("E13", "Sub", "none (provisional)", "all", "not recorded", "provisional rule, kept as final",
     "a district-energy revenue meter is a billing meter; nothing in CBECS shows interval data (by analogy with the utility-meter rule)"),
    ("E14", "Run+Load", "BAS trend logs", "row condition (chiller + BAS cooling)", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E15", "Run+Load", "BAS trend logs", "row condition (boiler + BAS heating)", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E16", "SpC", "BAS trend logs", "row condition", "reference-set linkage imputation", "", "as for the BAS rows of the code-clause table"),
    ("E17", "Lgt", "BAS trend logs (provisional)", BAS_LIGHT, "reference-set linkage imputation", "provisional rule, kept as final",
     "demand-responsive lighting is networked, but whether it stores a series is unknown; provisional: recorded only through a BAS that controls lighting"),
    ("E17", "Lgt", "none (provisional)", "otherwise", "not recorded", "provisional rule, kept as final", "as above"),
    ("E18", "T+Sp+Run", "BAS trend logs (provisional)", BAS_HVAC, "reference-set linkage imputation", "provisional rule, kept as final",
     "a VRF central controller may log locally or not at all; provisional: recorded only through a BAS"),
    ("E18", "T+Sp+Run", "none (provisional)", "otherwise", "not recorded", "provisional rule, kept as final", "as above"),
    ("E19", "Pwr", "BAS trend logs", "row condition", "reference-set linkage imputation", "", "weak row, sensitivity only"),
    ("E20", "Sub", "none (provisional)", "all", "not recorded", "provisional rule, kept as final",
     "targets can be tracked from bills alone; weak row, sensitivity only"),
]


def build_tables() -> None:
    rows = ami_rows()
    with open(config.TABLES / "L_ami_states.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(config.TABLES / "L_recording.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_row", "classes", "route", "condition", "rule", "flag", "evidence"])
        w.writerows(ROUTES)
    A = sum(r["ami_commercial"] for r in rows)
    T = sum(r["meters_commercial"] for r in rows)
    print(f"national commercial AMI {A:,} / {T:,} = {A / T:.4f}; states {len(rows)}")


# ── per-building route masks (static over draws) ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def routes(controls: str) -> dict:
    """Masks per building (int64 arrays) for the deterministic parts of each route, given the Table E reading."""
    from atlas import draws
    return routes_df(draws.stock().df, controls)


def routes_df(df: pd.DataFrame, controls: str) -> dict:
    """As ``routes`` for any frame with the survey's columns (each imputation's redrawn answers)."""
    from atlas import draws
    ev = lambda e: df.eval(e).fillna(False).astype(bool).to_numpy()
    bas_hvac, bas_light, smart = ev(BAS_HVAC), ev(BAS_LIGHT), (df.SMRTTHRM == 1).to_numpy()
    n = len(df)
    m = lambda s: draws.to_mask(s.split("+"))
    out = {"bas_hvac": bas_hvac, "bas_light": bas_light, "smart": smart,
           "E_bas": np.zeros(n, np.int64), "E_rd": np.zeros(n, np.int64), "E_flag_bas": np.zeros(n, np.int64),
           "E_flag_all": np.zeros(n, np.int64), "E_smart": np.zeros(n, np.int64), "E_occ_any": np.zeros(n, np.int64)}
    tags = declared.E_READINGS[controls]
    for row in declared.table_e():
        if row["reading"] not in tags or row["streams"] == "-" or row["id"] in declared.E_EXCLUDE.get(controls, ()):
            continue
        cond = ev(row["condition"])
        b = m(row["streams"])
        rid = row["id"]
        if rid in ("E01", "E02", "E09", "E10", "E14", "E15", "E16", "E19"):
            out["E_bas"] |= np.where(cond, b, 0)
        elif rid in ("E03", "E08"):
            out["E_bas"] |= np.where(cond & bas_light, b, 0)
            if rid == "E03":                    # optimistic: occupancy sensors recorded without BAS lighting control
                out["E_occ_any"] |= np.where(cond, b, 0)
        elif rid == "E04":
            out["E_smart"] |= np.where(cond, b, 0)
        elif rid in ("E07", "E12", "E18"):
            out["E_flag_bas"] |= np.where(cond & bas_hvac, b, 0)
            out["E_flag_all"] |= np.where(cond, b, 0)
        elif rid == "E17":
            out["E_flag_bas"] |= np.where(cond & bas_light, b, 0)
            out["E_flag_all"] |= np.where(cond, b, 0)
        elif rid in ("E13", "E20"):
            out["E_flag_all"] |= np.where(cond, b, 0)
        else:
            raise ValueError(f"no recording route for {rid}")
    return out


def ami_probs(state_idx: np.ndarray, year: int = 2018, measure: str = "ami") -> np.ndarray:
    """Probability that a building's whole-building meter is recorded as interval data: its state's commercial
    advanced-meter share (primary), or the state's share of commercial customers with daily digital access."""
    from atlas import draws
    states, _ = draws.state_rung_table()
    sh = ami_share(year) if measure == "ami" else daily_share(year)
    p = np.array([sh[s] for s in states])
    return p[state_idx]


def build_ami_year(year: int) -> None:
    """A later year's commercial advanced-meter shares (sensitivity S12): tables/L_ami_states_<year>.csv."""
    rows = ami_rows(year)
    with open(ami_table(year), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    A = sum(r["ami_commercial"] for r in rows)
    T = sum(r["meters_commercial"] for r in rows)
    print(f"{year}: national commercial AMI {A:,} / {T:,} = {A / T:.4f}; states {len(rows)}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["tables"]:
        build_tables()
    elif sys.argv[1:2] == ["ami"]:
        build_ami_year(int(sys.argv[2]))
    elif sys.argv[1:2] == ["daily"]:
        build_daily_year(int(sys.argv[2]))
    else:
        raise SystemExit("usage: python -m atlas.recording tables | ami <year> | daily <year>")
