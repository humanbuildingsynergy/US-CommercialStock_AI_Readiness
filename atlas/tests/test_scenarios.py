"""Description on non-BAS routes (design) and per building (variant) leave the primary unchanged at d = 1;
description outside the BAS can only raise the linked level against the all-routes rule; adoption lag only moves
editions down; the 90.1-only monitoring rule only removes meter-clause applicability; the code-family column and the
daily-access shares are well formed."""
import numpy as np
import pandas as pd
import pytest

from atlas import config, draws, floors, mi, recording


@pytest.fixture(scope="module")
def imp():
    return mi.make_imp("primary", 0)


def _timed(**tm):
    draws.TIMING.update(lag=tm.get("lag", 0), monitoring_901_only=tm.get("monitoring_901_only", False))
    try:
        return mi.make_imp("primary", 0)
    finally:
        draws.TIMING.update(lag=0, monitoring_901_only=False)


@pytest.mark.parametrize("scope,unit", [("nonbas", "stream"), ("nonbas", "building"), ("all", "stream"), ("all", "building")])
def test_d_one_is_the_primary_under_every_structure(imp, scope, unit):
    a = mi.evaluate(imp, mi.Cfg("primary"))
    b = mi.evaluate(imp, mi.Cfg("primary", d=1.0, desc_scope=scope, desc_unit=unit))
    assert np.array_equal(a["C"], b["C"])


def test_nonbas_description_is_at_least_the_all_routes_rule(imp):
    a = mi.evaluate(imp, mi.Cfg("primary", d=0.5, desc_scope="all"))
    b = mi.evaluate(imp, mi.Cfg("primary", d=0.5, desc_scope="nonbas"))
    assert np.all((a["C"] & ~b["C"]) == 0)
    assert b["sC"][:, 0].mean() > a["sC"][:, 0].mean()


def test_lag_moves_editions_down_only(imp):
    for lag in (1, 2):
        lagged = _timed(lag=lag)
        assert np.all(lagged.new_rung <= imp.new_rung)
        assert np.any(lagged.new_rung < imp.new_rung)


def test_monitoring_901_only_removes_meter_clause_applicability(imp):
    strict = _timed(monitoring_901_only=True)
    meter_ids = {r["id"] for r in floors.table_c() if r["reading"] == "meter_clause"}
    assert meter_ids
    fewer = 0
    for rid, (ok, _, _) in imp.rows.items():
        ok2 = strict.rows[rid][0]
        if rid in meter_ids:
            assert np.all(ok2 <= ok)
            fewer += int(np.sum(ok & ~ok2))
        else:
            assert np.array_equal(ok, ok2)
    assert fewer > 0


def test_code_family_column():
    V = pd.read_csv(config.TABLES / "V_states.csv")
    assert set(V.code_family.dropna()) <= {"90.1", "IECC", "state", "none", "implicit"}
    assert V.code_family.notna().all()


@pytest.mark.parametrize("year", [2018])
def test_daily_access_shares_are_shares(year):
    sh = recording.daily_share(year)
    v = np.array(list(sh.values()), float)
    assert np.all((v >= 0) & (v <= 1))
