"""The copied canon reproduces the readiness framework; the code-clause table's levels follow their stated rules."""
import re

import pytest

from atlas import canon, cbecs, config, floors, vintage

FRAMEWORK_CANON = config.SOURCES / "readiness" / "canon.py"   # the framework's canon module, release v1.0.0 (fetched)


def _block(text: str, name: str) -> str:
    m = re.search(rf"^{name} = _rows\((.*?)^\)", text, re.S | re.M)
    return re.sub(r"\s+", "", m.group(1))


@pytest.mark.skipif(not FRAMEWORK_CANON.exists(), reason="the framework's canon module is not fetched")
def test_canon_is_a_verbatim_copy_of_the_framework_release():
    """Read (never import) the framework's canon module and compare the requirement table."""
    ours = (config.REPO / "atlas" / "canon.py").read_text(encoding="utf-8")
    theirs = FRAMEWORK_CANON.read_text(encoding="utf-8")
    assert _block(ours, "_V2") == _block(theirs, "_V2")


def test_canon_total():
    assert canon.CANONS["v2"].total == 76


def test_published_ceilings():
    """The framework's published ceilings: minimal 7, comfort 12, standard 20 of 76; its metered offices 49 and its small
    offices 37 of 76. The 2004 code profile (zone temperature, setpoint, whole-building meter) scores 20 of 76."""
    sets = ({"T"}, {"T", "Sp"}, {"T", "Sp", "Occ"}, {"T", "Sp", "Whole"},
            {"T", "Sp", "Whole", "Occ", "Lgt", "Pwr", "Sub", "Sub2"}, {"T", "Sp", "Whole", "Occ", "Lgt", "Sub"})
    assert [canon.ceiling(s).achieved for s in sets] == [7, 12, 20, 20, 49, 37]


@pytest.fixture(scope="module")
def F():
    return floors.floors()


def test_levels_are_ordered(F):
    for rd in floors.READINGS:
        assert (F[f"{rd}_low"] <= F[f"{rd}_central"] + 1e-12).all()
        assert (F[f"{rd}_central"] <= F[f"{rd}_high"] + 1e-12).all()


def test_sensitivity_rows_never_enter_a_level(F):
    for c in F.columns:
        if c.endswith("streams"):
            assert not F[c].str.contains("CO2|OAT|SAT").any(), c


def test_main_is_contained_in_early_occ(F):
    for lv in floors.LEVELS:
        assert (F[f"main_{lv}"] <= F[f"early_occ_{lv}"] + 1e-12).all()


def test_conditions_drive_the_base_level(F):
    df = cbecs.load()
    uncond = ~((df.HEATP.fillna(0) > 0) | df.COOLP.notna())
    assert not F.loc[uncond, "main_central_streams"].str.contains("T").any()
    assert (F.loc[df.ELUSED == 2, "main_central"] == 0).all()


def test_monitoring_only_for_new_large_buildings_at_2013(F):
    df = cbecs.load()
    rungs = vintage.assign(df)
    has = F["main_high_streams"].str.contains("Sub2")
    assert (df.loc[has, "SQFT"] >= 25000).all()
    assert (rungs.loc[has, "new_high"] >= vintage.ladder().index("2013")).all()
