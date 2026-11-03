"""One entry point for the whole pipeline.

    python -m atlas fetch                 download the third-party inputs and check their SHA-256 (scripts/fetch_data.py)
    python -m atlas tables                rebuild the input tables derived from fetched files and the reference set
    python -m atlas analysis [--m M]      the multiple imputation, scenarios and sensitivities (results/)
    python -m atlas figures               the figures and table files under figures/
    python -m atlas ledger                results/ledger.md: every number the paper cites
    python -m atlas all [--quick] [--m M] fetch, tables, analysis, figures and ledger in order

--quick runs M = 10 imputations and a Sobol design of N = 512 (a few minutes on a laptop); the paper uses M = 200 and
N = 8,192. ATLAS_WORKERS sets the number of worker processes (default: the CPU count, at most 56); results do not depend
on it. Each step runs in its own process, in the order below, and stops the pipeline if it fails.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

TABLES = [["atlas.sources"], ["atlas.linkage"], ["atlas.recording", "tables"], ["atlas.recording", "ami", "2025"],
          ["atlas.recording", "daily", "2018"]]
ANALYSIS = [["atlas.assumption_check"], ["atlas.estimates", "{M}"], ["atlas.gap", "{M}"], ["atlas.cuts", "{M}"],
            ["atlas.sensitivities", "{M}"], ["atlas.scoring", "{M}"], ["atlas.refset_counts"],
            ["atlas.size_linkage"], ["atlas.editions", "{M}"], ["atlas.scenarios", "{M}"], ["atlas.weighting", "{M}"]]
FIGURES = ["fig_distribution", "fig_editions", "fig_gap", "fig_geography", "fig_lever", "fig_segments",
           "fig_sobol", "fig_tipping", "tables_v2", "table_sensitivities", "appendix_tables"]


def _run(cmd: list[str], env: dict) -> None:
    t0 = time.time()
    print(f"== {' '.join(cmd)}", flush=True)
    r = subprocess.run([PY, *cmd], cwd=ROOT, env=env)
    if r.returncode:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}")
    print(f"   done in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m atlas", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("fetch", "tables", "analysis", "figures", "ledger", "all"))
    ap.add_argument("--m", type=int, default=None, help="imputations per scenario (default 200; 10 with --quick)")
    ap.add_argument("--quick", action="store_true", help="M = 10 and a Sobol design of N = 512")
    a = ap.parse_args()
    M = a.m or (10 if a.quick else 200)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    if a.quick:
        env.setdefault("ATLAS_SOBOL_N", "512")
    t0 = time.time()
    steps = ("fetch", "tables", "analysis", "figures", "ledger") if a.step == "all" else (a.step,)
    for step in steps:
        if step == "fetch":
            _run(["scripts/fetch_data.py"], env)
        elif step == "tables":
            for c in TABLES:
                _run(["-m", *c], env)
        elif step == "analysis":
            for c in ANALYSIS:
                _run(["-m", *[x.format(M=M) for x in c]], env)
        elif step == "figures":
            for f in FIGURES:
                _run([f"figures/src/{f}.py"], env)
        elif step == "ledger":
            _run(["scripts/build_ledger.py"], env)
    print(f"finished {a.step} (M = {M}) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
