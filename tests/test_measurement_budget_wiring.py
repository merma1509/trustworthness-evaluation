"""Regression guards for the cost/budget wiring (RQ4 + Trust Budget).

These tests protect the pipeline connection that a fresh ``make run`` depends on:

1. The validation report must NOT over-sell an estimated/emulated human timing as
   a genuine "MEASURED" study. Only an explicit ``measurement_validity == "MEASURED"``
   marker counts; ``ESTIMATED_FROM_RECORDS`` and ``PLACEHOLDER_EMULATED`` must be
   labelled ESTIMATED so the cost ratio claim stays honest.

2. The budget optimizer must consume ``results/human_timing_measurement.json`` as
   the single source of truth (falling back to the 30 s placeholder when absent),
   so ``budget_plan.json`` and ``validation_report.json`` never diverge silently.
"""
import json
from pathlib import Path

import pytest

from scripts.budget_optimizer import (
    DEFAULT_HUMAN_SECONDS_PER_LABEL,
    HUMAN_SECONDS_PER_LABEL,
    HUMAN_TIMING_PATH,
    _load_human_seconds_per_label,
)
from src.validation import compute_measurement_budget

GATES = {"trust": 0.7, "unverified": 0.4}


def _validation_report():
    return {
        "rq1_agreement": {
            "by_dimension": {
                "safety": {"cohens_kappa": 1.0, "n": 10},
                "truthfulness": {"cohens_kappa": 0.0, "n": 10},
                "consistency": {"cohens_kappa": 1.0, "n": 10},
            }
        }
    }


# ── RQ4 validity classification ────────────────────────────
def _budget(validity, median=6.81):
    return compute_measurement_budget(
        auto_time_per_prompt_seconds=4.51,
        num_prompts=105,
        num_models=2,
        measured_human_timing={
            "median_seconds_per_label": median,
            "n_measured": 30,
            "measurement_validity": validity,
        },
    )


@pytest.mark.parametrize("validity,expected_measured", [
    ("ESTIMATED_FROM_RECORDS", False),   # reproducible estimate, NOT a live run
    ("PLACEHOLDER_EMULATED", False),     # demo placeholder
    (None, False),                       # no marker at all -> not measured
    ("", False),
    ("MEASURED", True),                  # only a genuine interactive study counts
])
def test_measurement_budget_validity_classification(validity, expected_measured):
    """Only an explicit 'MEASURED' marker may be sold as a measured cost ratio."""
    cost = _budget(validity)
    assert cost["cost_is_measured"] is expected_measured


def test_estimate_is_not_over_sold():
    """ESTIMATED_FROM_RECORDS → cost_basis must say ESTIMATE, not MEASURED."""
    cost = _budget("ESTIMATED_FROM_RECORDS")
    assert cost["cost_is_measured"] is False
    assert cost["cost_basis"].upper().startswith("ESTIMATE")


def test_measured_marker_flips_basis_to_measured():
    cost = _budget("MEASURED")
    assert cost["cost_is_measured"] is True
    assert cost["cost_basis"].upper().startswith("MEASURED")


def test_ratio_drops_when_human_time_used():
    """A shorter real/estimated per-label time gives a smaller (honest) ratio."""
    est = compute_measurement_budget(
        auto_time_per_prompt_seconds=4.51,
        num_prompts=105,
        num_models=2,
        measured_human_timing={
            "median_seconds_per_label": 6.81,
            "measurement_validity": "ESTIMATED_FROM_RECORDS",
        },
    )
    placeholder = compute_measurement_budget(
        auto_time_per_prompt_seconds=4.51,
        num_prompts=105,
        num_models=2,
    )
    assert est["cost_ratio_estimated_x"] < placeholder["cost_ratio_estimated_x"]


# ── Budget optimizer single source of truth ─────────────────
def test_budget_plan_records_timing_source_and_rate():
    """budget_plan.json must expose the timing rate AND where it came from."""
    from scripts.budget_optimizer import build_plan

    plan = build_plan(_validation_report(), GATES, source="test")
    # The value used must match what the validation report would use.
    assert plan["human_seconds_per_label"] == HUMAN_SECONDS_PER_LABEL
    # Provenance must be recorded so a reviewer can see it's from the file.
    assert isinstance(plan["human_timing_source"], str)
    assert "human_timing_measurement.json" in plan["human_timing_source"]


def test_loader_falls_back_to_placeholder_when_file_absent(tmp_path, monkeypatch):
    """If the timing file is missing, fall back to the 30 s placeholder."""
    from scripts import budget_optimizer as bo

    missing = tmp_path / "human_timing_measurement.json"
    monkeypatch.setattr(bo, "HUMAN_TIMING_PATH", missing)
    # Re-read the module-level constant logic via the helper against a missing path.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bo, "HUMAN_TIMING_PATH", tmp_path / "does_not_exist.json")
        assert bo._load_human_seconds_per_label() == DEFAULT_HUMAN_SECONDS_PER_LABEL


def test_loader_reads_median_from_real_file(tmp_path):
    """When the file exists, its median_seconds_per_label is used verbatim."""
    from scripts import budget_optimizer as bo

    timing_file = tmp_path / "human_timing_measurement.json"
    timing_file.write_text(
        json.dumps({"median_seconds_per_label": 7.25}), encoding="utf-8"
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bo, "HUMAN_TIMING_PATH", timing_file)
        assert bo._load_human_seconds_per_label() == 7.25
