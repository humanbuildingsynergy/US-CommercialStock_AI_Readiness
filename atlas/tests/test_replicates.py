"""The jackknife reproduces EIA's worked example (2018 microdata user's guide, PROC SURVEYFREQ):
SQFTC = 2 has 2,832,884 buildings with standard error 139,667 (RSE 4.93)."""
import pytest

from atlas import cbecs, replicates


def test_users_guide_example():
    df = cbecs.load(replicates=True)
    est, se = replicates.estimate(df, lambda d, w: w[d.SQFTC == 2].sum())
    assert est == pytest.approx(2_832_884, abs=2)
    assert se == pytest.approx(139_667, abs=2)
    assert round(100 * se / est, 2) == 4.93
