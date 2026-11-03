"""Predictive tail probabilities for the model check."""
import numpy as np

from atlas import estimates


def test_tails():
    sim = np.array([0, 1, 2, 3, 6, 6, np.nan, 4, 5, 2], float)
    t = estimates.tails(sim, 6)
    assert t["Pr(sim >= obs)"] == 2 / 9 and t["Pr(sim <= obs)"] == 1.0
    assert t["two-sided p"] == 4 / 9
    assert estimates.tails(np.array([1.0, 1.0]), 1)["two-sided p"] == 1.0     # capped at 1
