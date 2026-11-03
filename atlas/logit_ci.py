"""95 % intervals for shares on the logit scale.

A Rubin's-rules interval for a share can cross 0 or 1 (e.g. a status share of 0.8 % [-0.2 %, 1.8 %]). For every
statistic that is a share, the interval is built on the logit scale and transformed back:

    logit(Q̄) ± t_ν · √T / (Q̄ (1 − Q̄))   →   expit(·)

the first-order (delta-method) form of pooling logit(Q_m) with its within-imputation variance U_m / (Q_m (1 − Q_m))²:
B and W scale by the same factor, so the fraction of missing information and the Barnard-Rubin degrees of freedom are
unchanged, and the pooled point estimate Q̄ stays on the natural scale. A share that is exactly 0 or 1 in every
imputation (T = 0) keeps its degenerate interval. Means, ratios, gains, differences and Shapley contributions stay on the
natural scale.

Applied after each runner writes its tables (and once over the committed estimates tables):

    python -m atlas.logit_ci [dir ...]     default: results/estimates
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from atlas import config

ALL_SHARES = {"editions_imputed.csv", "zero_profile.csv"}          # every statistic is a share of floor area
NOT_SHARES = {"decomposition_scenarios.csv", "h5_lever.csv", "m_stability.csv", "sobol.csv",
              "scenario_tipping.csv", "transfer_tipping.csv", "model_checks.csv", "donor_cells.csv",
              "growth.csv"}
NATURAL = ("mean ", "gain", "drop", "minus", "ratio", "H2b", "H4", "size:", "change", "Shapley")


def is_share(file: str, statistic: str) -> bool:
    if file in ALL_SHARES:
        return True
    s = str(statistic)
    return not (any(k in s for k in NATURAL) or s.endswith(" level"))   # "linked level" is a mean; "linked level zero" a share


def vocab(text: str) -> str:
    """Vocabulary for generated text: the readiness framework's 45 knowledge graphs are the "reference set", never
    "the census" (census division stays); "advanced meter" for AMI."""
    import re
    t = re.sub(r"\b[Cc]ensus(?= (?:knowledge|graphs?|frequenc|parameters|linkage|rates?|shares?|odds|test|bound|bootstrap|log-odds|"
               r"buildings?|counts?|types?|statistics?|value|model|check))", "reference-set", text)
    t = re.sub(r"\bcensus-to-stock\b", "reference-set-to-stock", t)
    t = re.sub(r"\bcensus-supported\b", "reference-set-supported", t)
    t = re.sub(r"\b[Tt]he census\b(?! division)", lambda m: m.group(0)[0] + "he reference set", t)
    t = re.sub(r"\bAMI-recorded\b", "advanced-meter-recorded", t)
    return t


def is_ratio(file: str, statistic: str) -> bool:
    """The H2b concentration ratio (a ratio of shares) takes its interval on the log scale."""
    s = str(statistic)
    return s.startswith("H2b") and "size" not in s


def log_interval(est, T, df):
    est, T, df = (np.asarray(x, dtype=float) for x in (est, T, df))
    lo, hi = est.copy(), est.copy()
    ok = (T > 0) & (est > 0)
    t = stats.t.ppf(0.975, np.maximum(df[ok], 1))
    h = t * np.sqrt(T[ok]) / est[ok]
    lo[ok], hi[ok] = est[ok] * np.exp(-h), est[ok] * np.exp(h)
    return lo, hi


def logit_interval(est: np.ndarray, T: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est, T, df = (np.asarray(x, dtype=float) for x in (est, T, df))
    lo, hi = est.copy(), est.copy()
    ok = (T > 0) & (est > 0) & (est < 1)
    t = stats.t.ppf(0.975, np.maximum(df[ok], 1))
    c = np.log(est[ok] / (1 - est[ok]))
    h = t * np.sqrt(T[ok]) / (est[ok] * (1 - est[ok]))
    lo[ok], hi[ok] = 1 / (1 + np.exp(-(c - h))), 1 / (1 + np.exp(-(c + h)))
    return lo, hi


def apply_frame(D: pd.DataFrame, file: str) -> pd.DataFrame:
    if file in NOT_SHARES or not {"estimate", "T", "df", "lo95", "hi95", "statistic"} <= set(D.columns):
        return D
    m = np.array([is_share(file, s) for s in D.statistic])
    r = np.array([is_ratio(file, s) for s in D.statistic])
    if "part" in D.columns:                               # changes and Shapley parts are differences
        m &= ~D.part.isin(["change"]).to_numpy()
        r &= ~D.part.isin(["change"]).to_numpy()
    if "group" in D.columns and file == "reads_decomposition.csv":   # only the donor-redraw contrast (shares) and H2b (a ratio)
        m &= (D.group == "imputed-answer contrast").to_numpy()
        r = (D.group == "h2b").to_numpy()
    if not (m.any() or r.any()):
        return D
    D = D.copy()
    if m.any():
        lo, hi = logit_interval(D.loc[m, "estimate"], D.loc[m, "T"], D.loc[m, "df"])
        D.loc[m, "lo95"], D.loc[m, "hi95"] = lo, hi
    if r.any():
        lo, hi = log_interval(D.loc[r, "estimate"], D.loc[r, "T"], D.loc[r, "df"])
        D.loc[r, "lo95"], D.loc[r, "hi95"] = lo, hi
    D["interval scale"] = np.where(m, "logit", np.where(r, "log", "natural"))
    return D


def apply_file(path: Path) -> bool:
    D = pd.read_csv(path)
    if "interval scale" in D.columns:                     # idempotent: recompute from estimate, T and df
        D = D.drop(columns="interval scale")
    E = apply_frame(D, path.name)
    if E is D:
        return False
    E.to_csv(path, index=False, float_format="%.6g")
    return True


def apply_dir(d: Path) -> list[str]:
    return [p.name for p in sorted(Path(d).glob("*.csv")) if apply_file(p)]


if __name__ == "__main__":
    import sys
    dirs = [Path(a) for a in sys.argv[1:]] or [config.RESULTS / "estimates"]
    for d in dirs:
        if d.exists():
            print(d.name, apply_dir(d))
