"""Rubin's rules reduce to the jackknife when the imputation models are degenerate, and seeds reproduce."""
import numpy as np
import pytest

from atlas import mi


def _q_u(imps, stat="mean linked level"):
    Q, U = [], []
    for imp in imps:
        ev = mi.evaluate(imp, mi.Cfg("primary"))
        q, u, names = mi.jackknife(mi.columns(ev, full=False), mi.weights()["area"])
        i = names.index(stat)
        Q.append([q[i]])
        U.append([u[i]])
    return np.array(Q), np.array(U)


def test_degenerate_imputation_gives_jackknife_variance():
    imps = [mi.make_imp("primary", m, degenerate=True) for m in range(3)]
    Q, U = _q_u(imps)
    R = mi.rubin(Q, U)
    assert R.B.iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert R["T"].iloc[0] == pytest.approx(U[0, 0], rel=1e-12)


def test_rubin_rules_formulae():
    Q = np.array([[1.0], [2.0], [3.0]])
    U = np.array([[0.5], [0.5], [0.5]])
    R = mi.rubin(Q, U)
    assert R.estimate.iloc[0] == 2.0 and R.B.iloc[0] == 1.0 and R["T"].iloc[0] == pytest.approx(0.5 + 4 / 3)


def test_mi_seeds_reproduce():
    a = mi.evaluate(mi.make_imp("primary", 7), mi.Cfg("primary"))
    b = mi.evaluate(mi.make_imp("primary", 7), mi.Cfg("primary"))
    assert np.array_equal(a["C"], b["C"]) and np.array_equal(a["F"], b["F"])


def test_reference_grid_point_equals_primary():
    imp = mi.make_imp("primary", 3)
    a = mi.evaluate(imp, mi.Cfg("primary"))
    b = mi.evaluate(imp, mi.Cfg("primary", c=1.0, d=1.0, da=0.0, db=0.0))
    assert np.array_equal(a["C"], b["C"])
