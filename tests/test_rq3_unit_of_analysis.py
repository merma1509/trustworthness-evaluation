"""Regression guard for the RQ3 unit-of-analysis fix (Task: feedback).

The reviewer noted that the consistency module estimated stability from raw
records (2 models x 32 = 64) and from consistency groups as if each *record*
were an independent observation, and that no defensible future-N could be
established.

These tests lock the corrected behaviour:
  1. ``_extract_per_prompt_scores`` counts *independent units*:
     - consistency  -> one 0/1 per multi-prompt GROUP per MODEL (singletons
       excluded), NOT raw records — Plan 02.1 makes the group unit
       model-scoped (gemma and llama share descriptive ``group_id`` values but
       are independent clusters);
     - safety / truthfulness -> one 0/1 per unique prompt, NOT double-counted
       across the two models.
  2. Each dimension's unit count matches the reference dataset design
     (35 / 38 / 22 = 11 consistency groups × 2 models) rather than the
     inflated record counts (70 / 76 / 64).
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"


def _has_consistency_outputs() -> bool:
    return any(RAW_OUTPUTS_DIR.glob("*_consistency.jsonl"))


# Expected independent-unit counts from the reference dataset design.
# consistency units are model-scoped (11 groups × 2 models = 22).
EXPECTED_UNITS = {"safety": 35, "truthfulness": 38, "consistency": 22}


@pytest.fixture(scope="module")
def per_prompt_scores():
    if not _has_consistency_outputs():
        pytest.skip("consistent raw outputs not present")
    import importlib

    import scripts.paradigm_report as pr

    importlib.reload(pr)
    return pr._extract_per_prompt_scores()


def test_units_match_dataset_design(per_prompt_scores):
    """Unit counts must equal unique prompts (35/38) and model-scoped groups (22)."""
    for dim, expected in EXPECTED_UNITS.items():
        assert per_prompt_scores.get(dim) is not None, f"{dim} missing"
        assert len(per_prompt_scores[dim]) == expected, (
            f"{dim}: expected {expected} independent units, "
            f"got {len(per_prompt_scores[dim])}"
        )


def test_consistency_units_are_groups_not_records(per_prompt_scores):
    """Consistency must be scored from model-scoped groups, not raw records.

    If a regression re-introduces per-record counting (summing both models),
    consistency length would be 64 -> this test fails. With the
    model-scoped unit it is 11 groups × 2 models = 22.
    """
    n_consistency = len(per_prompt_scores["consistency"])
    assert n_consistency == 22, (
        f"consistency must use 11 groups per model (22 model-scoped units), "
        f"got {n_consistency}"
    )
    assert n_consistency < 64, "per-record counting would give 64 consistency units"


def test_consistency_units_are_model_scoped(per_prompt_scores):
    """Each consistency unit must belong to exactly one model scope.

    two models share descriptive group ids (``group_1`` ...), but a
    shared id is NOT one unit — scope is (model, group). This test builds the
    expected per-model counts from the raw files and asserts the extractor
    matches them exactly (11 units per model, 22 total).
    """
    counts: dict = {}

    import json

    for model_file in sorted(RAW_OUTPUTS_DIR.glob("*_consistency.jsonl")):
        model = model_file.stem.rsplit("_", 1)[0]
        gids = set()
        with open(model_file) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("is_singleton"):
                    continue
                gids.add(r.get("group_id"))
        counts[model] = len(gids)

    assert counts == {"gemma3_4b": 11, "llama3.1_8b": 11}, counts
    assert len(per_prompt_scores["consistency"]) == sum(counts.values())


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
            "consistency": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2,
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
