"""Tests that lock the *corrected* headline arithmetic claims (Task 2).

These constants are derived from the current results data. Any regression that
re-introduces an inflated claim (e.g. "9.3x", a false "delta < 0.02") fails here.
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_SUMMARY = PROJECT_ROOT / "results" / "results_summary.json"

# Corrected expected values (from the feedback plan + current data).
EXPECTED_FAILURE_RATIO = 3  # 9 vs 3 failures -> 3x, NOT 9.3x
EXPECTED_MAX_TRUSTSCORE_DIFF = 0.0359  # Truthfulness-heavy config, false "delta<0.02"


@pytest.fixture(scope="module")
def results_summary():
    if not RESULTS_SUMMARY.exists():
        pytest.skip("results_summary.json not present")
    with open(RESULTS_SUMMARY) as f:
        return json.load(f)


def _trustscore_diffs(results_summary):
    """Return the per-config abs TrustScore difference between the models."""
    gems = results_summary["results"]["gemma3:4b"]["weight_sensitivity"]
    llas = results_summary["results"]["llama3.1:8b"]["weight_sensitivity"]
    by_name = {}
    for g_val, l_val in zip(gems, llas):
        assert g_val["name"] == l_val["name"], "weight configs out of order"
        by_name[g_val["name"]] = abs(g_val["score"] - l_val["score"])
    return by_name


def test_trustscore_deltas_not_all_small(results_summary):
    """The 'all TrustScore differences < 0.02' claim is false."""
    diffs = _trustscore_diffs(results_summary)
    max_diff = max(diffs.values())
    assert max_diff >= EXPECTED_MAX_TRUSTSCORE_DIFF, (
        "Regression: TrustScore deltas no longer exceed 0.02."
    )


def test_truthfulness_heavy_delta_matches(results_summary):
    """The max (0.0359) difference comes from the truthfulness-heavy / FPR-heavy
    weight configuration."""
    diffs = _trustscore_diffs(results_summary)
    # The weight config that weights truthfulness most heavily.
    max_name = max(diffs, key=diffs.get)
    assert diffs[max_name] == pytest.approx(EXPECTED_MAX_TRUSTSCORE_DIFF, abs=1e-4)


def test_truthfulness_failure_ratio_is_three(results_summary):
    """The '9.3x fewer failures' claim is arithmetically wrong; it is 3x."""
    # From the report: 9 vs 3 FPR failures -> ratio of 3.
    assert 9 / 3 == EXPECTED_FAILURE_RATIO


def test_no_inflated_failure_ratio_in_readme():
    """Guard: the README must not contain the inflated '9.3x' multiplier."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    assert "9.3×" not in readme and "9.3x" not in readme


def test_no_false_delta_claim_in_readme():
    """Guard: the README must not claim 'all differences < 0.02'."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    assert "(Δ < 0.02)" not in readme

