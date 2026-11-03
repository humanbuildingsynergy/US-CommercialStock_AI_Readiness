"""Stage 1, Table V: the edition assignment follows its stated rule."""
import pytest

from atlas import cbecs, vintage as v


@pytest.fixture(scope="module")
def rec():
    df = cbecs.load()
    return df, v.assign(df)


def test_ladder_and_publication_order():
    L = v.ladder()
    assert L[0] == "pre-1989" and L[-1] == "2022"
    years = [v.published()[e] for e in L]
    assert years == sorted(years)


def test_every_division_has_states_and_all_states_are_present():
    assert v.divisions() == list(range(1, 10))
    assert sum(len(v.division_states(d)) for d in v.divisions()) == 51   # 50 states + DC


def test_state_step_function_reads_the_table():
    # PNNL-31437 Table A.1: Connecticut 90.1-2007 (IECC 2009) in 2012, 90.1-2010 in 2016
    assert v.state_edition_at("CT", 2011) == "2004"
    assert v.state_edition_at("CT", 2012) == "2007"
    assert v.state_edition_at("CT", 2016) == "2010"
    assert v.state_edition_at("AK", 1990) == "pre-1989"


def test_division_is_lower_median_over_states():
    # New England at 2015 by hand: CT 2007, ME 2007, NH 2007, MA 2010, RI 2010, VT 2013 -> 2007
    by_hand = sorted(v.ladder().index(v.state_edition_at(s, 2015)) for s in v.division_states(1))
    assert [v.ladder()[i] for i in by_hand] == ["2007", "2007", "2007", "2010", "2010", "2013"]
    assert v.edition_at(1, 2015) == "2007"


def test_edition_never_decreases_with_year():
    L = v.ladder()
    for d in v.divisions():
        seq = [L.index(v.edition_at(d, y)) for y in range(1980, 2019)]
        assert seq == sorted(seq), d


def test_interval_is_plus_minus_one_rung_and_clamped():
    for r in v.table_v().itertuples():
        lo, c, hi = (v.ladder().index(x) for x in (r.low, r.central, r.high))
        assert c - 1 <= lo <= c <= hi <= c + 1
        end = int(r.years.split("-")[-1]) if "-" in r.years else int(r.years.lstrip("<")) - 1
        assert v.published()[r.high] <= end


def test_no_building_beyond_2016(rec):
    _, a = rec
    assert a.filter(like="_high").max().max() <= v.ladder().index("2016")


def test_renovation_only_moves_systems_up_and_only_named_systems(rec):
    df, a = rec
    tv = v.table_v().set_index(["basis", "yrconc", "cendiv"])
    L = v.ladder()
    for pubid, r in df.iterrows():
        base = L.index(tv.loc[("construction", int(r.YRCONC), int(r.CENDIV)), "central"])
        for var, (sysname,) in v.renovation_rules().items():
            got = a.at[pubid, f"{sysname}_central"]
            assert got >= base
            if r.RENOV != 1 or r[var] != 1:
                assert got == base, (pubid, var)


def test_2013_2018_buildings_have_no_renovation_row():
    tv = v.table_v()
    assert not ((tv.basis == "renovation") & (tv.yrconc == 9)).any()
