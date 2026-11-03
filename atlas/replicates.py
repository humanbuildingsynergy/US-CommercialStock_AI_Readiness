"""CBECS 2018 jackknife standard errors (EIA, "User's Guide to the 2018 CBECS Public Use Microdata
File", pinned in INPUTS.md): Var(θ) = α Σ_r (θ_r − θ)², with R = 151 replicate weights FINALWT1..151
and α = 1. ``estimate(df, f)`` evaluates a statistic f(df, weights) at FINALWT and each replicate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

R, ALPHA = 151, 1.0
REPLICATES = tuple(f"FINALWT{i}" for i in range(1, R + 1))


def estimate(df: pd.DataFrame, stat: Callable[[pd.DataFrame, pd.Series], float]) -> tuple[float, float]:
    """(full-sample estimate, jackknife standard error)."""
    theta = stat(df, df["FINALWT"])
    reps = np.array([stat(df, df[w]) for w in REPLICATES])
    return float(theta), float(np.sqrt(ALPHA * ((reps - theta) ** 2).sum()))
