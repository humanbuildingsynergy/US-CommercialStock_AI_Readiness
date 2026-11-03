"""Sensitivity S13: DOE's dates replace PNNL's only when the variant is on, and the primary table is restored after."""
from atlas import vintage


def test_doe_dates_variant_switches_and_restores():
    base = vintage.state_edition_at("OH", 2020)
    vintage.configure(doe_dates=True)
    try:
        assert vintage.state_edition_at("OH", 2020) != base      # DOE: no 90.1-2013 or 2016 in Ohio before 2024
        assert vintage.state_edition_at("OH", 2024) in ("2016", "2019")
    finally:
        vintage.configure()
    assert vintage.state_edition_at("OH", 2020) == base
