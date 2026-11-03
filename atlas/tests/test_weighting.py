"""The primary and HVAC-credit scores are unchanged by the scoring refactor; the domain- and intent-equal
scorings change C(I) only (tier coverages identical); the stream-group Shapley shares add up to C(I) under every
scoring; the canon's weights are as specified."""
import numpy as np
import pytest

from atlas import bounds, canon, draws, tiers

STREAMS = tuple(canon.STREAMS)


def _sets(n=300, seed=14):
    rng = np.random.default_rng(seed)
    out = [frozenset(), frozenset(STREAMS)]
    for _ in range(n):
        out.append(frozenset(s for s in STREAMS if rng.random() < rng.uniform(0.1, 0.8)))
    return out


def test_canon_weights():
    c = canon.CANONS["v2"]
    by_domain = {d: sum(len(r) for (dd, _), r in c.rungs.items() if dd == d) for d in canon.DOMAINS}
    assert by_domain == {"thermal": 16, "energy": 12, "airquality": 16, "equipment": 16, "occupancy": 8, "lighting": 8}
    assert c.total == 76


@pytest.mark.parametrize("scoring,can", [("primary", "v2"), ("hvac", "v2h")])
def test_primary_and_hvac_scores_unchanged(scoring, can):
    for s in _sets():
        assert canon.readiness(s, scoring) == canon.ceiling(s, can).score


@pytest.mark.parametrize("scoring", ["domain_equal", "intent_equal"])
def test_equal_scorings_change_readiness_only(scoring):
    moved = 0
    for s in _sets():
        r = canon.readiness(s, scoring)
        assert 0.0 <= r <= 1.0
        for t in tiers.TIERS:
            assert tiers.coverage(s, t, scoring) == tiers.coverage(s, t)
        moved += abs(r - canon.readiness(s)) > 1e-12
    assert canon.readiness(frozenset(STREAMS), scoring) == pytest.approx(1.0)
    assert moved > 0


@pytest.mark.parametrize("scoring", ["primary", "domain_equal", "intent_equal"])
def test_group_shares_add_up_to_readiness(scoring):
    for s in _sets(60):
        assert sum(bounds._phi(s, scoring)) == pytest.approx(canon.readiness(s, scoring), abs=1e-12)


def test_score_table_under_domain_equal_tiers():
    m = np.array([draws.to_mask(s) for s in _sets(40)], dtype=np.int64)
    a, b = draws.score(m, "domain_equal"), draws.score(m, "domain_equal_tiers")
    assert np.array_equal(a[:, 0], b[:, 0])
    assert np.all((b[:, 1:] >= 0) & (b[:, 1:] <= 1))
