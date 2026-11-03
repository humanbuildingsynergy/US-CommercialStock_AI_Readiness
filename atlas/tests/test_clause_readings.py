"""§8.4.3 exception f (hotels, motels, restaurants), C06 held out of the primary result (S24), not-used rows, C15 (§10.4.5)
credited on a compliance stream of its own."""
from dataclasses import replace

from atlas import draws, floors, mi


def test_c08_exempts_hotels_motels_restaurants():
    c08 = next(r for r in floors.table_c() if r["id"] == "C08")
    df = draws.stock().df
    ok = floors._condition(df, c08["condition"]).to_numpy()
    ex = df.PBAPLUS.isin(floors.EXEMPT_843).to_numpy()
    assert not (ok & ex).any() and (ok == ((df.ELUSED == 1) & (df.SQFT >= 25000)).to_numpy() & ~ex).all()
    assert not (mi.clause_scope() & ex).any() and (mi.clause_scope() == ((df.SQFT >= 25000).to_numpy() & ~ex)).all()


def test_c06_held_in_the_imputation_applied_only_under_s24():
    assert "occ_alt" not in floors.READINGS["main"] and "occ_alt" in floors.READINGS["alt_lighting"]
    assert mi.held_ids() == {"C06"}
    imp = mi.make_imp("primary", 0)
    assert "C06" in imp.rows                       # drawn: the random stream is the one C06 was part of
    a, b = mi.evaluate(imp, mi.Cfg()), mi.evaluate(imp, mi.Cfg(alt_lighting=True))
    c06 = imp.rows["C06"][0]
    assert ((a["F"] != b["F"]) <= c06).all()       # S24 differs only where C06 applies
    assert ((a["F"] & ~b["F"]) == 0).all()         # and only adds


def test_not_used_rows_enter_no_reading():
    tags = frozenset().union(*floors.READINGS.values(), *floors.HELD.values())
    rows = [r for r in floors.table_c() if r["reading"] == "not_used"]
    assert [r["id"] for r in rows] == ["C16", "C17", "C18"]
    assert "not_used" not in tags


def test_c15_condition_has_no_hotel_exception():
    c15 = next(r for r in floors.table_c() if r["id"] == "C15")
    assert c15["reading"] == "meter_clause" and c15["streams"] == "Sub" and c15["rung_from"] == "2013"
    df = draws.stock().df
    ok = floors._condition(df, c15["condition"]).to_numpy()
    assert (ok == ((df.SQFT >= 25000) & ((df.CHWT == 1) | (df.STHW == 1))).to_numpy()).all()
    assert (ok & df.PBAPLUS.isin(floors.EXEMPT_843).to_numpy()).any()   # hotels, motels, restaurants not exempt


def test_c15_logs_no_electricity():
    assert "C15" not in mi.elec_log_ids() and "C08" in mi.elec_log_ids()


def test_c15_own_stream_moves_only_its_buildings():
    for m in (0, 1):
        imp = mi.make_imp("primary", m)
        assert "C15" in imp.rows and imp.rows["C15"][2]
        without = replace(imp, rows={k: v for k, v in imp.rows.items() if k != "C15"})
        c15 = imp.rows["C15"][0]
        for cfg in (mi.Cfg(), mi.Cfg(c=0.5)):
            a, b = mi.evaluate(imp, cfg), mi.evaluate(without, cfg)
            for k in ("F", "D", "C"):
                assert ((a[k] != b[k]) <= c15).all()       # only buildings meeting C15's condition change
                assert ((b[k] & ~a[k]) == 0).all()          # and only gain
        s = draws.to_mask(("Sub",))
        a = mi.evaluate(imp, mi.Cfg())
        assert ((a["F"] & s) > 0)[c15].mean() > 0.5
