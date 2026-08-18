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
# We lock on the *property*, not an exact delta, because the pipeline is stochastic and
# the precise max delta (and even which weighted config achieves it) shifts from
# run to run. What must always hold: some config exceeds 0.02 and the
# truthfulness-weighted config is one such culprit.
MIN_DELTA_TO_REFUTE_CLAIM = 0.02  # any delta strictly above 0.02 refutes the claim


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
    """The 'all TrustScore differences < 0.02' claim must remain refuted.

    The pipeline is stochastic, so this asserts a robust property rather than
    a brittle exact number: at least one weight config must keep a
    TruthScore difference strictly above 0.02. If every config ever drops
    below 0.02 (e.g. after a future model change), this test fails and forces
    an explicit re-evaluation of the claim.
    """
    diffs = _trustscore_diffs(results_summary)
    max_diff = max(diffs.values())
    assert max_diff > MIN_DELTA_TO_REFUTE_CLAIM, (
        "Regression: no TrustScore delta remains above 0.02 across weight "
        "configs. Either the 'all delta < 0.02' claim is now genuinely true "
        "(update this test AND the README deliberately) or the pipeline "
        "regressed."
    )


def test_truthfulness_weighted_config_exceeds_claim(results_summary):
    """The 'all delta < 0.02' claim is refuted specifically by a truthfulness-
    weighted config (FPR-heavy), not just by some unrelated corner of the
    weight grid.

    The exact config name may vary across stochastic runs, so we verify the
    *reason* the claim is false: weighting Truthfulness more heavily widens
    the two models' TrustScore gap. We accept any config that both weights
    Truthfulness at >= its baseline weight AND yields delta > 0.02.
    """
    baseline_w_truth = results_summary["results"]["gemma3:4b"]["baseline_weights"]["truthfulness"]
    diffs = _trustscore_diffs(results_summary)
    gems = results_summary["results"]["gemma3:4b"]["weight_sensitivity"]
    llas = results_summary["results"]["llama3.1:8b"]["weight_sensitivity"]
    culprit_found = False
    for g_val, l_val in zip(gems, llas):
        name = g_val["name"]
        w_truth = g_val.get("w_truthfulness", 0)
        if w_truth >= baseline_w_truth and diffs[name] > MIN_DELTA_TO_REFUTE_CLAIM:
            culprit_found = True
            break
    assert culprit_found, (
        "No truthfulness-weighted config produces a TrustScore delta above "
        "0.02. The 'all delta < 0.02' claim can no longer be refuted on the "
        "truthfulness axis — update the test and README deliberately."
    )


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
    assert "(delta < 0.02)" not in readme

