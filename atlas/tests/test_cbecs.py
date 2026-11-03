"""Stage 1: the pinned microdata loads, its codes match the codebook, and the weights reproduce
EIA's published Table B1."""
import pandas as pd
import pytest

from atlas import cbecs, codebook, config, distributions


@pytest.fixture(scope="module")
def df():
    return cbecs.load()


def test_codebook_parses_split_rows():
    v = codebook.variables()
    assert v["PBAPLUS"].codes["29"] == "High school"        # "29=High school 30=..." on one line
    assert v["PBAPLUS"].codes["30"] == "Preschool/daycare"
    assert v["EMCS"].codes == {"1": "Yes", "2": "No", "Missing": "Not applicable"}
    assert list(v["YRCONC"].codes) == [str(c) for c in range(2, 10)]


def test_codes_in_file_are_codebook_codes(df):
    v = codebook.variables()
    for var in cbecs.CATEGORICAL:
        seen = {str(int(c)) for c in df[var].dropna().unique()}
        assert seen <= set(v[var].codes), (var, seen - set(v[var].codes))


def test_pinned_and_complete(df):
    assert len(df) == config.N_RECORDS
    assert df.FINALWT.gt(0).all()


def test_weights_reproduce_table_b1(df):
    """Buildings exactly; floor area within 1 M ft2, because the public file top-codes SQFT at
    2,100,000 ft2 (20 records) while B1 is tabulated from the unmasked areas: the total and the
    over-500,000 row come out 0.8 M ft2 (0.001 %) low, every other row exact."""
    area = df.FINALWT * df.SQFT
    close = lambda got, want: (round(got[0]) == want[0]) and abs(got[1] - want[1]) < 1.0
    assert round(df.FINALWT.sum() / 1e3) == config.B1_TOTAL["buildings_k"]
    assert abs(area.sum() / 1e6 - config.B1_TOTAL["sqft_m"]) < 1.0
    for code, (b, a) in config.B1_BY_YRCONC.items():
        m = df.YRCONC == code
        assert close((df.FINALWT[m].sum() / 1e3, area[m].sum() / 1e6), (b, a)), code
    for code, (b, a) in config.B1_BY_SQFTC.items():
        m = df.SQFTC.isin(code if isinstance(code, tuple) else (code,))
        assert close((df.FINALWT[m].sum() / 1e3, area[m].sum() / 1e6), (b, a)), code


def test_skip_universes_hold(df):
    assert all(cbecs.check_universes(df).values())


def test_pbaplus_nests_in_pba(df):
    fam = cbecs.family_of_pbaplus(df)
    assert fam[2] == 2 and fam[42] == 25 and fam[50] == 23   # office, retail, strip mall


def test_distributions_sum_to_stock(df):
    for var in ("YRCONC", "EMCS_SCOPE", "LTCTL"):
        t = distributions.table(df, var)
        assert t.n.sum() == config.N_RECORDS
        assert t.pct_area.sum() == pytest.approx(100.0)
