"""The HVAC-credit scoring never scores a stream set lower than the question set, and credits run status for the
equipment status question; the partial-graph class rates follow the Method; the optimistic switches only add streams."""
import itertools

import numpy as np

from atlas import canon, mi, tiers
from atlas.mi import Cfg


def test_hvac_credit_never_lower_and_credits_run_status():
    pool = ("T", "Sp", "SAT", "Run", "Load", "Pwr", "Whole", "Occ")
    for k in range(0, 5):
        for s in itertools.combinations(pool, k):
            s = set(s)
            assert canon.ceiling(s, "v2h").achieved >= canon.ceiling(s).achieved
            for t in tiers.TIERS:
                assert tiers.coverage(frozenset(s), t, "hvac") >= tiers.coverage(frozenset(s), t) - 1e-12
    assert canon.level("equipment", "status", {"Run"}, "hvac") == 1 and canon.level("equipment", "status", {"Run"}) == 0


def test_partial_rates_come_from_partial_graphs():
    dec, lin, ty = mi._census_arrays()
    types, q = mi.draw_params(np.random.default_rng(0), degenerate=True)
    k = mi.CLS.index("Pwr")                     # 5 partial declarations, 1 linked: empirical 0.2 (overall ratio was 11/21)
    assert abs(q[k] - 0.2) < 1e-12


def test_optimistic_switches_only_add():
    imp = mi.make_imp("enriched (high)", 0, extra_controls=("optimistic",))
    base = mi.evaluate(imp, Cfg("enriched (high)"))
    opt = mi.evaluate(imp, Cfg("enriched (high)", controls="optimistic", occ_any=True, big_meter=True))
    assert ((base["C"] & ~opt["C"]) == 0).all()          # same uniforms: every linked stream stays linked
