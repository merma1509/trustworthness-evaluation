"""Tests for the Budget-vs-Reliability curve helper (Part-3 Figure 3)."""
import pytest

from scripts.budget_reliability_curve import reliability_gain


def test_no_labels_no_gain():
    """Spending zero labels leaves reliability at the starting κ."""
    assert reliability_gain(0.615, 0, 50) == pytest.approx(0.615)


def test_full_coverage_approaches_cap():
    """Labelling the entire dimension lifts reliability most of the way to 1.0.

    The proxy saturates asymptotically (share/(share+k_sat)), so full coverage
    gets close to the cap without necessarily reaching 1.0 exactly.
    """
    assert 0.75 < reliability_gain(0.0, 50, 50) < 1.0


def test_monotonic_in_labels():
    """More labels never reduce reliability."""
    prev = reliability_gain(0.3, 0, 60)
    for lab in (5, 10, 20, 40, 60):
        cur = reliability_gain(0.3, lab, 60)
        assert cur >= prev
        prev = cur


def test_low_kappa_has_more_headroom():
    """A brittle dimension (κ=0) gains more per label than a robust one (κ=0.6)."""
    brittle = reliability_gain(0.0, 6, 30) - reliability_gain(0.0, 0, 30)
    robust = reliability_gain(0.6, 6, 30) - reliability_gain(0.6, 0, 30)
    assert brittle > robust


def test_diminishing_returns_saturate():
    """Marginal gain per extra label decreases as labels accumulate."""
    # 0->10 vs 50->60 (same +10 step) on a 100-record brittle dimension.
    early = reliability_gain(0.0, 10, 100) - reliability_gain(0.0, 0, 100)
    late = reliability_gain(0.0, 60, 100) - reliability_gain(0.0, 50, 100)
    assert early > late > 0
