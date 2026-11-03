"""The reference-set graphs that link none, treated as missing (sensitivities S21 and S22)."""
import numpy as np

from atlas import link_none, mi


def test_audit_counts():
    assert link_none.none_graphs() == ("bldg12", "bldg31", "bldg35", "bldg37", "bldg39", "bldg41", "bldg42", "bldg43", "bldg44")
    assert set(link_none.no_reference_graphs()) == set(link_none.none_graphs()) - {"bldg31", "bldg35"}


def test_drop_refits_types_only():
    t0, q0 = mi.draw_params(np.random.default_rng(0), degenerate=True)
    t21, q21 = mi.draw_params(np.random.default_rng(0), degenerate=True, drop=link_none.none_graphs())
    t22, q22 = mi.draw_params(np.random.default_rng(0), degenerate=True, drop=link_none.no_reference_graphs())
    assert np.allclose(t0, np.array([22, 9, 6]) / 37)
    assert np.allclose(t21, np.array([22, 0, 6]) / 28)
    assert np.allclose(t22, np.array([22, 2, 6]) / 30)
    # class rates come from the partial graphs, which neither variant drops
    assert np.allclose(q0, q21, equal_nan=True) and np.allclose(q0, q22, equal_nan=True)


def test_drop_keeps_the_random_stream():
    # same number of draws whatever is dropped, so a sensitivity differs from the primary only in what it varies
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    mi.draw_params(a)
    mi.draw_params(b, drop=link_none.none_graphs())
    assert a.random() == b.random()
