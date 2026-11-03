"""Stage 3: the census copy is faithful, r follows its definition, the three numbers per record
obey floor <= central <= declared, and tiers follow their definition."""
import pytest

from atlas import canon, cbecs, declared, tiers


def test_copied_census_recomputes_with_copied_canon():
    c = declared.census()
    assert len(c) == 45
    for r in c.itertuples():
        d = set() if r.streams_declared == "-" else set(r.streams_declared.split("+"))
        k = set() if r.streams_linked == "-" else set(r.streams_linked.split("+"))
        assert canon.ceiling(d).score == pytest.approx(r.c_ans_declared, abs=1e-3), r.building
        assert canon.ceiling(k).score == pytest.approx(r.c_ans_linked, abs=1e-3), r.building


def test_discount_matches_the_framework_census_facts():
    r = declared.discount()
    assert r["n_declare_more"] == 15                     # the readiness framework: 15 of 45
    assert r["n_linked_series"] == 30
    assert r["R2_low"] < r["R1"] < r["R2_high"] < 1
    assert r["R1"] < r["R3"]


def test_primary_configuration():
    assert declared.PRIMARY == {"floor": "main", "controls": "primary", "r": "R1"}
    assert "wo_bas_occ" not in declared.E_READINGS["primary"]        # E02 is a sensitivity


@pytest.fixture(scope="module")
def R():
    return declared.records()


def test_floor_central_declared_are_ordered(R):
    for m in ("C", *tiers.TIERS):
        for lv in ("low", "central", "high"):
            f, c, d = R[f"floor_{m}_{lv}"], R[f"central_{m}_{lv}"], R[f"declared_{m}_{lv}"]
            assert (f <= c + 1e-12).all() and (c <= d + 1e-12).all(), (m, lv)


def test_primary_contains_wo_without_e02():
    df = cbecs.load()
    wo = declared.declared_streams(df, "wo")
    pr = declared.declared_streams(df, "primary")
    e02_only = df.eval("EMCS==1 & EMCSLT==1 & OCSN!=1").fillna(False)
    for i in df.index[~e02_only]:
        assert wo[i] <= pr[i]


def test_tier_coverage_sums_to_ceiling():
    """Per-intent coverages, weighted by each intent's caps, recompose C(I)."""
    c = canon.CANONS["v2"]
    caps = {i: sum(len(r) for (d, ii), r in c.rungs.items() if ii == i) for i in canon.INTENTS}
    for s in ({"T", "Sp", "Whole"}, {"T", "Sp", "Whole", "Occ", "Lgt", "CO2"}):
        fs = frozenset(s)
        total = sum(tiers.coverage(fs, i) * caps[i] for i in canon.INTENTS)
        assert total == canon.ceiling(fs).achieved


def test_bas_row_adds_nothing_on_the_76_cell_canon():
    """Structural fact: T+Sp+SAT+Run over the base floor scores as the floor."""
    base = {"T", "Sp", "Whole"}
    assert canon.ceiling(base | {"SAT", "Run"}).score == canon.ceiling(base).score


def test_smart_thermostat_row_only_fires_without_a_bas():
    df = cbecs.load()
    assert (df.loc[df.SMRTTHRM == 1, "EMCS"] == 2).all()


def test_enriched_reading_contains_primary_and_is_not_a_fixed_set():
    """Enriched rows add to the primary reading; the alternatives fixed before the analysis exclude them."""
    df = cbecs.load()
    pr = declared.declared_streams(df, "primary")
    en = declared.declared_streams(df, "enriched")
    assert all(pr[i] <= en[i] for i in df.index)
    assert "enriched" not in declared.PRE_REGISTERED_E
    # E12: every building reporting an economizer declares outdoor-air temperature
    assert all("OAT" in en[i] for i in df.index[df.ECN == 1])
    # E16 is the only source of the scheduled setpoint
    no = declared.declared_streams(df, "enriched_no_spc")
    assert not any("SpC" in no[i] for i in df.index)
