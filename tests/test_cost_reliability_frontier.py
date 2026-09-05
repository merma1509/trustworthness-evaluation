"""Tests for scripts/cost_reliability_frontier.py (Part 3C §3C.5.3).

Guards the key framing fix: the cost-reliability frontier is a **MODEL
ESTIMATE**, never a measured result. The output must:
- Be explicitly labelled as a MODEL estimate.
- Expose the assumed human accuracy as an assumption.
- Not bury the modelling nature of the effective-agreement curve.
"""

import pytest

from scripts.cost_reliability_frontier import build_frontier

KAPPAS = {"safety": 0.593, "truthfulness": 0.277, "consistency": 0.0}


def test_output_is_marked_model():
    frontier = build_frontier(KAPPAS)
    assert "MODEL" in frontier["label"].upper()
    assert "not measured" in frontier["label"].lower()


def test_frontier_has_one_point_per_fraction():
    for dim, f in build_frontier(KAPPAS)["frontiers"].items():
        assert len(f["points"]) == 5  # 0, 25, 50, 75, 100%
        for p in f["points"]:
            assert p["label"] == "MODEL ESTIMATE (not measured)"


def test_effective_agreement_starts_at_auto_kappa():
    """At 0% human, effective agreement == (modelled) auto κ."""
    f = build_frontier(KAPPAS)["frontiers"]
    for dim, k in KAPPAS.items():
        assert f[dim]["points"][0]["effective_agreement_model"] == pytest.approx(k)


def test_effective_agreement_rises_with_human_fraction():
    f = build_frontier(KAPPAS)["frontiers"]
    for dim in KAPPAS:
        pts = f[dim]["points"]
        for a, b in zip(pts, pts[1:]):
            assert b["effective_agreement_model"] >= a["effective_agreement_model"]


def test_cost_grows_with_fraction():
    f = build_frontier(KAPPAS)["frontiers"]
    for dim in KAPPAS:
        pts = f[dim]["points"]
        assert pts[4]["cost_usd"] > pts[0]["cost_usd"]
        # 100% human cost == n * sec/label * rate.
        assert pts[4]["cost_usd"] == pytest.approx(
            f[dim]["n_responses"] * 12.2 / 3600 * 20.0, rel=1e-2
        )


def test_human_accuracy_is_an_explicit_assumption():
    frontier = build_frontier(KAPPAS)
    assert frontier["human_accuracy_assumed"] == 0.9
    # And it appears in each dim block too.
    for dim in KAPPAS:
        assert frontier["frontiers"][dim]["human_accuracy_assumed"] == 0.9
