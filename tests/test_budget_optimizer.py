"""Tests for the κ-gated Trust Budget optimizer (Part 3 §4.2)."""

import pytest

from scripts.budget_optimizer import build_plan

GATES = {"trust": 0.7, "unverified": 0.4}


def _sealed_report():
    """Minimal mimic of scripts/report_agreement.py's report schema."""
    return {
        "gold_vs_auto": {
            "per_dimension": {
                "safety": {"n": 16, "cohens_kappa": 0.9, "agreement_rate": 0.9},
                "truthfulness": {"n": 24, "cohens_kappa": 0.55, "agreement_rate": 0.79},
                "consistency": {"n": 23, "cohens_kappa": 0.3, "agreement_rate": 0.82},
            },
            "overall": {"n": 63, "cohens_kappa": 0.655, "agreement_rate": 0.746},
        }
    }


def _validation_report():
    """Minimal mimic of results/validation_report.json (RQ1 by_dimension)."""
    return {
        "rq1_agreement": {
            "by_dimension": {
                "safety": {"cohens_kappa": 0.6154, "n": 10},
                "truthfulness": {"cohens_kappa": 0.0, "n": 10},
                "consistency": {"cohens_kappa": 0.6154, "n": 10},
            }
        }
    }


@pytest.mark.parametrize("factory,expected_band", [
    (_sealed_report,
     {"safety": "trust", "truthfulness": "caveated", "consistency": "unverified"}),
    (_validation_report,
     {"safety": "caveated", "truthfulness": "unverified", "consistency": "caveated"}),
])
def test_band_assignment(factory, expected_band):
    """Each dimension lands in the band implied by its κ and the gates."""
    plan = build_plan(factory(), GATES, source="test")
    by_dim = {r["dimension"]: r["band"] for r in plan["by_dimension"]}
    assert by_dim == expected_band


def test_unverified_routes_all_records_to_humans():
    """κ < unverified gate → every record in that dimension needs annotation."""
    plan = build_plan(_validation_report(), GATES, source="test")
    truth = next(r for r in plan["by_dimension"] if r["dimension"] == "truthfulness")
    assert truth["band"] == "unverified"
    assert truth["annotations_needed"] == 10  # full dimension (n=10)


def test_trust_needs_no_human_budget():
    """κ ≥ trust gate → zero annotations allocated."""
    plan = build_plan(_sealed_report(), GATES, source="test")
    safety = next(r for r in plan["by_dimension"] if r["dimension"] == "safety")
    assert safety["band"] == "trust"
    assert safety["annotations_needed"] == 0


def test_caveated_spot_checks_ten_percent():
    """caveated band → ~10% spot-check, never the whole dimension."""
    plan = build_plan(_sealed_report(), GATES, source="test")
    truth = next(r for r in plan["by_dimension"] if r["dimension"] == "truthfulness")
    assert truth["band"] == "caveated"
    assert truth["annotations_needed"] == 2  # round(24 * 0.10)


def test_missing_kappa_defaults_to_sampling():
    """A dimension with no κ estimate is flagged as 'unknown', not assumed trusted."""
    report = _sealed_report()
    # Remove the gold-vs-auto entry for consistency entirely.
    report["gold_vs_auto"]["per_dimension"]["consistency"] = {}
    plan = build_plan(report, GATES, source="test")
    cons = next(r for r in plan["by_dimension"] if r["dimension"] == "consistency")
    assert cons["band"] == "unknown"
    assert cons["recommendation"].startswith("No κ estimate")


def test_small_sample_high_kappa_is_downgraded_to_caveated():
    """A high κ estimated from a tiny sample must NOT land in 'trust'.

    With n=3 the CI around κ is essentially the whole range; the optimizer
    downgrades to 'caveated' and allocates a spot-check instead of zero."""
    report = {
        "gold_vs_auto": {
            "per_dimension": {
                "safety": {"n": 3, "cohens_kappa": 0.95, "agreement_rate": 1.0},
            },
            "overall": {"n": 3, "cohens_kappa": 0.95, "agreement_rate": 1.0},
        }
    }
    plan = build_plan(report, {"trust": 0.7, "unverified": 0.4}, source="test")
    safety = next(r for r in plan["by_dimension"] if r["dimension"] == "safety")
    assert safety["band"] == "caveated"
    assert safety["annotations_needed"] == 1  # round(3 * 0.10)
    assert "too wide to trust" in safety["recommendation"]


def test_adequate_sample_high_kappa_stays_trust():
    """With a sufficient n, a high κ is trusted (no human budget)."""
    report = {
        "gold_vs_auto": {
            "per_dimension": {
                "safety": {"n": 40, "cohens_kappa": 0.9, "agreement_rate": 0.95},
            },
            "overall": {"n": 40, "cohens_kappa": 0.9, "agreement_rate": 0.95},
        }
    }
    plan = build_plan(report, {"trust": 0.7, "unverified": 0.4}, source="test")
    safety = next(r for r in plan["by_dimension"] if r["dimension"] == "safety")
    assert safety["band"] == "trust"
    assert safety["annotations_needed"] == 0


def test_total_cost_non_negative():
    """Cost is the sum of allocated labels at the measured rate."""
    plan = build_plan(_validation_report(), GATES, source="test")
    assert plan["total_human_annotations"] >= 0
    assert plan["estimated_human_cost_usd"] >= 0
