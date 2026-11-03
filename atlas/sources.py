"""Tables rebuilt from fetched third-party files (``scripts/fetch_data.py``), each checked against the committed copy.

  tables/R_reference_set.csv   the readiness framework's census of 45 real building knowledge graphs (declared and
                               linked stream classes, ceilings), from its release file results/census/census_v2.json
                               (v1.0.0; licensed CC BY-NC 4.0, see NOTICE)
  tables/V_state_weights.csv   2018 commercial-sector energy consumption by state (EIA SEDS use_all_btu.csv, MSN TECCB),
                               the weight of a state within its census division

    python -m atlas.sources          -> rebuild both and report whether they equal the committed tables
"""
from __future__ import annotations

import csv
import io
import json

from atlas import config

READINESS = config.SOURCES / "readiness"
SEDS = config.SOURCES / "seds"


def reference_set_rows() -> list[dict]:
    d = json.loads((READINESS / "census_v2.json").read_text(encoding="utf-8"))
    return [{"source": r["source"], "building": r["building"], "points": r["points"], "linked_points": r["linked_points"],
             "streams_declared": "+".join(r["streams_declared"]) or "-", "streams_linked": "+".join(r["streams"]) or "-",
             "c_ans_declared": round(r["c_ans_declared"], 4), "c_ans_linked": round(r["c_ans"], 4)} for r in d["rows"]]


def state_weight_rows() -> list[dict]:
    src = "EIA SEDS use_all_btu.csv, MSN TECCB (total energy consumption in the commercial sector), 2018"
    with open(SEDS / "use_all_btu.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["MSN"] == "TECCB" and r["State"] != "US"]
    return [{"state": r["State"], "teccb_2018_billion_btu": int(float(r["2018"])), "source": src}
            for r in sorted(rows, key=lambda r: r["State"])]


def _text(rows: list[dict], lineterminator: str) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator=lineterminator)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def rebuild(name: str, rows: list[dict], write: bool = False) -> bool:
    """True when the rebuilt table equals the committed one (line endings aside); writes it when *write*."""
    p = config.TABLES / name
    old = p.read_bytes().decode("utf-8") if p.exists() else ""
    nl = "\r\n" if "\r\n" in old else "\n"
    new = _text(rows, nl)
    same = new.replace("\r\n", "\n") == old.replace("\r\n", "\n")
    if write and not same:
        p.write_text(new, encoding="utf-8", newline="")
    return same


def main(write: bool = False) -> dict[str, bool]:
    out = {"R_reference_set.csv": rebuild("R_reference_set.csv", reference_set_rows(), write),
           "V_state_weights.csv": rebuild("V_state_weights.csv", state_weight_rows(), write)}
    for k, v in out.items():
        print(f"{k}: {'identical to the committed table' if v else 'DIFFERS from the committed table'}")
    return out


if __name__ == "__main__":
    import sys
    main(write="--write" in sys.argv)
