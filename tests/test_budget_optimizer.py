"""Tests for the κ-gated Trust Budget optimizer (Part 3 §4.2)."""
import json

import pytest

from scripts.budget_optimizer import build_plan

GATES = {"trust": 0.7, "unverified": 0.4}


def _experiment_report():
    """Minimal mimic of scripts/run_blinded_annotation.py's report schema."""
    return {
        "by_dimension": {
            "safety": {
                "auto_comparison": {"cohens_kappa": 0.9, "n_valid_pairs": 45},
                "adjudicated": {"n_total": 52},
            },
            "truthfulness": {
                "auto_comparison": {"cohens_kappa": 0.55, "n_valid_pairs": 38},
                "adjudicated": {"n_total": 46},
            },
            "consistency": {
                "auto_comparison": {"cohens_kappa": 0.3, "n_valid_pairs": 17},
                "adjudicated": {"n_total": 20},
            },
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
    (_experiment_report,
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
    plan = build_plan(_experiment_report(), GATES, source="test")
    safety = next(r for r in plan["by_dimension"] if r["dimension"] == "safety")
    assert safety["band"] == "trust"
    assert safety["annotations_needed"] == 0


def test_caveated_spot_checks_ten_percent():
    """caveated band → ~10% spot-check, never the whole dimension."""
    plan = build_plan(_experiment_report(), GATES, source="test")
    truth = next(r for r in plan["by_dimension"] if r["dimension"] == "truthfulness")
    assert truth["band"] == "caveated"
    assert truth["annotations_needed"] == 5  # round(46 * 0.10)


def test_missing_kappa_defaults_to_sampling():
    """A dimension with no κ estimate is flagged as 'unknown', not assumed trusted."""
    report = _experiment_report()
    # Remove the auto-comparison for consistency entirely.
    report["by_dimension"]["consistency"] = {}
    plan = build_plan(report, GATES, source="test")
    cons = next(r for r in plan["by_dimension"] if r["dimension"] == "consistency")
    assert cons["band"] == "unknown"
    assert cons["recommendation"].startswith("No κ estimate")


def test_total_cost_non_negative():
    """Cost is the sum of allocated labels at the measured rate."""
    plan = build_plan(_validation_report(), GATES, source="test")
    assert plan["total_human_annotations"] >= 0
    assert plan["estimated_human_cost_usd"] >= 0
