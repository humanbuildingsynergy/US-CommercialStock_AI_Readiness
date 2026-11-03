"""Edition groups, the latest-edition variant and the domain jackknife."""
import numpy as np

from atlas import mi, vintage, editions


def test_every_ladder_rung_maps_to_one_of_the_six_groups():
    g = editions.group_of_rung()
    assert len(g) == len(vintage.ladder())
    assert set(g) <= set(range(len(editions.GROUPS)))
    lad = vintage.ladder()
    assert g[lad.index("pre-1989")] == 0 and g[lad.index("1989")] == 1 and g[lad.index("1999/2001")] == 2
    assert g[lad.index("2004")] == g[lad.index("2007")] == 3 and g[lad.index("2010")] == 4
    assert g[lad.index("2013")] == g[lad.index("2016")] == 5


def test_latest_edition_is_never_older_than_construction():
    imp = mi.make_imp("primary", 0)
    assert imp.latest_rung is not None and (imp.latest_rung >= imp.new_rung).all()
    assert (imp.latest_rung > imp.new_rung).any()      # some systems were renovated under a newer edition


def test_jackknife_survives_a_replicate_without_the_group():
    w = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 2.0, 2.0]])   # full weight + 2 replicates
    seg = np.array([1.0, 0.0, 0.0])                                      # replicate 2 drops the group's only record
    Q, U, _ = editions._jack({"x": np.array([0.5, 1.0, 0.0])}, w, seg)
    assert Q[0] == 0.5 and np.isfinite(U).all()
