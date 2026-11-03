"""The per-building draws reduce to the deterministic records when degenerate, variant A
preserves the stock mean, seeds reproduce, and the census linkage split is as stated."""
import numpy as np
import pytest

from atlas import declared, draws, linkage


def test_census_linkage_split():
    assert linkage.type_counts() == {"all": 22, "none": 9, "partial": 6}
    g = linkage.graphs()
    assert int(g.in_R1.sum()) == 37


def test_deterministic_sample_reproduces_records():
    st = draws.stock()
    R = declared.records(st.df)
    X = draws.deterministic("primary rows")["X"]
    for j, m in enumerate(("C", "status", "summary", "diagnosis")):
        assert np.array_equal(X[:, j], R[f"central_{m}_central"].to_numpy())


def test_vectorised_floor_equals_table_c_floor():
    F0, _, _ = draws.base("primary rows")
    assert np.array_equal(draws.floor_masks(draws.stock().rungs), F0)


def test_full_compliance_reproduces_floor():
    F0, _, D0 = draws.base("primary rows")
    rng = draws.seed_for("t")
    n = len(F0)
    comply = {r["id"]: np.ones(n, bool) for r in draws.floor_rows() if r["comply"]}
    assert np.array_equal(draws.floor_masks(draws.stock().rungs, comply), F0)
    X, info = draws.sample(draws.Spec("c1", c=1.0), rng)
    assert np.array_equal(info["F"], F0) and np.array_equal(info["D"], D0)


@pytest.mark.parametrize("ty,which", [("all", "declared"), ("none", "floor")])
def test_degenerate_linkage_type(monkeypatch, ty, which):
    monkeypatch.setattr(linkage, "type_probs_pop", lambda name="R1": {k: float(k == ty) for k in ("all", "none", "partial")})
    _, _, D0 = draws.base("primary rows")
    F0 = draws.base("primary rows")[0]
    X, _ = draws.sample(draws.Spec("deg", linkage="B"), draws.seed_for("deg"))
    want = draws.score(D0 if which == "declared" else F0)
    assert np.array_equal(X, want)


def test_variant_a_preserves_mean():
    det = draws.deterministic("primary rows")["mean C"][0]
    got = draws.run(draws.Spec("A-test", linkage="A"), n_draws=200)["mean C"][0]
    assert got == pytest.approx(det, abs=2e-3)


def test_uniform_graph_draw_does_not_preserve_mean():
    """The mean of per-graph ratios (0.672) is not R1 (0.627): the reason variant A weights by C_declared."""
    vals, p_w = draws._graph_ratios(weighted=True)
    _, p_u = draws._graph_ratios(weighted=False)
    assert (vals * p_w).sum() == pytest.approx(declared.discount()["R1"], abs=1e-12)
    assert (vals * p_u).sum() > declared.discount()["R1"] + 0.03


def test_seeds_reproduce():
    s = draws.Spec("seed-test", linkage="B", edition="weighted", c=0.75)
    a, b = draws.run(s, n_draws=20), draws.run(s, n_draws=20)
    assert np.array_equal(a["E"], b["E"])
    assert a["mean C"] == b["mean C"]
