"""Shares get logit-scale intervals inside (0, 1), the H2b ratio a log-scale one; degenerate shares
and means keep theirs."""
import numpy as np
import pandas as pd

from atlas import logit_ci


def test_share_interval_stays_inside_the_unit_interval_and_contains_the_estimate():
    lo, hi = logit_ci.logit_interval([0.008, 0.5, 0.95], [2.5e-5, 1e-3, 1e-4], [70, 70, 70])
    assert (lo > 0).all() and (hi < 1).all()
    assert (lo < [0.008, 0.5, 0.95]).all() and (hi > [0.008, 0.5, 0.95]).all()
    # at 0.5 the logit interval is symmetric and close to the natural one for a small T
    assert abs((0.5 - lo[1]) - (hi[1] - 0.5)) < 1e-12


def test_degenerate_shares_and_means_are_left_alone():
    D = pd.DataFrame({"statistic": ["status tier (linked)", "diagnosis tier (linked)", "mean linked level", "H2b concentration ratio"],
                      "estimate": [0.08, 0.0, 0.14, 4.3], "T": [1e-4, 0.0, 1e-4, 0.5], "df": [80, 80, 80, 80],
                      "lo95": [0.06, 0.0, 0.12, 2.9], "hi95": [0.10, 0.0, 0.16, 5.7]})
    E = logit_ci.apply_frame(D, "national.csv")
    assert list(E["interval scale"]) == ["logit", "logit", "natural", "log"]   # H2b on the log scale
    assert E.loc[1, "lo95"] == 0.0 and E.loc[1, "hi95"] == 0.0
    assert np.allclose(E.loc[2, ["lo95", "hi95"]].to_numpy(dtype=float), D.loc[2, ["lo95", "hi95"]].to_numpy(dtype=float))
    assert 0 < E.loc[3, "lo95"] < 4.3 < E.loc[3, "hi95"]
