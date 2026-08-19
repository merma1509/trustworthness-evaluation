"""Regression guard for the RQ3 unit-of-analysis fix (Task: feedback).

The reviewer noted that the consistency module estimated stability from raw
records (2 models x 32 = 64) and from consistency groups as if each *record*
were an independent observation, and that no defensible future-N could be
established.

These tests lock the corrected behaviour:
  1. ``_extract_per_prompt_scores`` counts *independent units*:
     - consistency  -> one 0/1 per multi-prompt GROUP (singletons excluded),
       NOT raw records;
     - safety / truthfulness -> one 0/1 per unique prompt, NOT double-counted
       across the two models.
  2. Each dimension's unit count matches the reference dataset design
     (35 / 38 / 11 groups) rather than the inflated record counts
     (70 / 76 / 64).
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"


def _has_consistency_outputs() -> bool:
    return any(RAW_OUTPUTS_DIR.glob("*_consistency.jsonl"))


# Expected independent-unit counts from the reference dataset design.
EXPECTED_UNITS = {"safety": 35, "truthfulness": 38, "consistency": 11}


@pytest.fixture(scope="module")
def per_prompt_scores():
    if not _has_consistency_outputs():
        pytest.skip("consistent raw outputs not present")
    import importlib

    import scripts.paradigm_report as pr

    importlib.reload(pr)
    return pr._extract_per_prompt_scores()


def test_units_match_dataset_design(per_prompt_scores):
    """Unit counts must equal unique prompts (35/38) and groups (11)."""
    for dim, expected in EXPECTED_UNITS.items():
        assert per_prompt_scores.get(dim) is not None, f"{dim} missing"
        assert len(per_prompt_scores[dim]) == expected, (
            f"{dim}: expected {expected} independent units, "
            f"got {len(per_prompt_scores[dim])}"
        )


def test_consistency_units_are_groups_not_records(per_prompt_scores):
    """Consistency must be scored from 11 groups, not 64 raw records.

    If a regression re-introduces per-record counting (summing both models),
    consistency length would be 64 -> this test fails.
    """
    n_consistency = len(per_prompt_scores["consistency"])
    assert n_consistency == 11, f"consistency must use 11 groups, got {n_consistency}"


def test_consistency_values_are_group_verdicts(per_prompt_scores):
    """Each consistency unit must be the group's boolean verdict (0/1)."""
    for v in per_prompt_scores["consistency"]:
        assert v in (0.0, 1.0)


def test_safety_truthfulness_not_double_counted(per_prompt_scores):
    """Safety/truthfulness must not sum both models (would be 70/76)."""
    assert len(per_prompt_scores["safety"]) == 35  # not 70
    assert len(per_prompt_scores["truthfulness"]) == 38  # not 76


def test_validation_report_uses_defensible_future_n():
    """The full validation report must expose a closed-form future-N estimate
    (replacing the old 'cannot establish future N' analysis)."""
    from src.validation import compute_validation_report

    report = compute_validation_report(
        audit_records=[],  # no human labels needed for RQ3
        per_prompt_scores={
            "safety": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] * 4,
            "truthfulness": [1.0, 1.0, 0.0, 1.0, 0.0] * 8,
            "consistency": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
    )
    rq3 = report["rq3_dataset_stability"]
    assert "consistency" in rq3
    cons = rq3["consistency"]
    # unit_of_analysis should flag the cluster nature of consistency.
    assert cons.get("unit_of_analysis") == "group (cluster)"
    # Both precision targets must be present and finite.
    assert cons["required_n_ci_width_0_10"]["n_required"] > 0
    assert cons["required_n_ci_width_0_05"]["n_required"] > 0
    # Tighter precision requires >= N.
    assert (
        cons["required_n_ci_width_0_05"]["n_required"]
        >= cons["required_n_ci_width_0_10"]["n_required"]
    )
