"""The reference-set knowledge graphs that link none of their declared streams, treated as missing data in sensitivities
S21 (all 9 dropped) and S22 (only those of them that publish no time-series reference at all dropped); both are rows of
atlas.sensitivities. This module names the two sets and audits the published graphs:

  tables/R_linkage_none_audit.csv   per graph that links none: whether its published graph carries any time-series
                                    reference, read from the Mortar graph files (each file's sha256 checked against
                                    tables/R_linkage_size.csv) and checked against the census's own count of linked points

The audit table is committed; ``audit`` rebuilds it from a folder of the Mortar graphs.

    python -m atlas.link_none <mortar dir>    -> tables/R_linkage_none_audit.csv
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from atlas import config, declared, linkage

AUDIT = config.TABLES / "R_linkage_none_audit.csv"
TS_ID = "ref:hasTimeseriesId"      # the only time-series predicate in the Mortar release (ref:hasExternalReference blocks)


def none_graphs() -> tuple[str, ...]:
    """The reference-set graphs (C_declared > 0) that link none of their declared streams."""
    g = linkage.graphs()
    return tuple(g[g.type == "none"].building)


@lru_cache(maxsize=1)
def no_reference_graphs() -> tuple[str, ...]:
    """Those of them whose published graph carries no time-series reference at all (from the committed audit)."""
    A = pd.read_csv(AUDIT)
    return tuple(A[A["any time-series reference"] == "no"].building)


def audit(mortar: Path) -> pd.DataFrame:
    g = linkage.graphs().set_index("building")
    cen = declared.census().set_index("building")
    sha = pd.read_csv(config.TABLES / "R_linkage_size.csv").set_index("building").file_sha256
    rows = []
    for b in none_graphs():
        f = Path(mortar) / f"{b}.ttl"
        raw = f.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        assert h == sha[b], f"{b}: file differs from the one the census and R_linkage_size.csv read"
        t = raw.decode("utf-8", errors="replace")
        stmts = [s for s in re.split(r"\s\.\s*\n", t) if TS_ID in s]
        n_ids = t.count(TS_ID)
        other = len(re.findall(r"time.?series", t, re.I)) - n_ids     # any other spelling of a time-series reference
        rows.append({
            "source": g.loc[b, "source"], "building": b, "points": int(cen.loc[b, "points"]),
            "any time-series reference": "yes" if n_ids or other else "no",
            "points with a time-series reference": len(stmts),
            "census linked points": int(cen.loc[b, "linked_points"]),
            "streams linked (census)": g.loc[b, "streams_linked"],
            "how checked": f"Mortar release file {b}.ttl (sha256 {h[:12]}…, as in R_linkage_size.csv): statements with "
                           f"{TS_ID} inside ref:hasExternalReference, the release's only time-series predicate "
                           f"({n_ids} ids; {max(other, 0)} other 'timeseries' mentions); agrees with the census's linked "
                           f"points" if len(stmts) == int(cen.loc[b, "linked_points"]) else "DISAGREES with the census"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    A = audit(Path(sys.argv[1]))
    A.to_csv(AUDIT, index=False)
    print(A.drop(columns="how checked").to_string())
