"""The Shapley decomposition behind H3 recomposes the full R²."""
import pytest

from atlas import cbecs, declared, stock


def test_shapley_recomposes_r2():
    df = cbecs.load(replicates=True)
    y = declared.records(df)["central_C_central"].to_numpy(dtype=float)
    out = stock.h4(df, y, jack=False)
    for t in out.values():
        parts = t.drop(index=["R2 full", "vintage - type"])["estimate"].sum()
        assert parts == pytest.approx(t.loc["R2 full", "estimate"], abs=1e-9)
