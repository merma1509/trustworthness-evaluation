"""Tests for scripts/compute_cost.py (Part 3C cost breakdown).

Guards the methodological fixes:
1. Every budget number is explicitly tagged MEASURED / ASSUMED / MODEL.
2. Setup/training overhead is surfaced (not silently ignored).
3. Adjudication cost is only counted when a disagreement rate is supplied,
   otherwise it is labelled as OMITTED rather than silently 0.
4. The nominal ratio is always accompanied by an apples-to-oranges caveat.
"""

import pytest

from scripts.compute_cost import compute_breakdown


def test_labels_are_explicit():
    """The returned structure must expose MEASURED / ASSUMED / MODEL buckets."""
    b = compute_breakdown(n_responses=210)
    assert "MEASURED" in b
    assert "ASSUMED" in b
    assert "MODEL" in b
    # Measured timing values are present.
    assert b["MEASURED"]["auto_seconds_per_prompt"] > 0
    assert b["MEASURED"]["human_seconds_per_label"] > 0
    # Setup hours appear under ASSUMED.
    assert b["ASSUMED"]["setup_hours"] > 0


def test_setup_included_by_default():
    """Setup/training cost must be part of the human total by default."""
    b = compute_breakdown(n_responses=210, human_hr_rate=20.0)
    assert b["human"]["setup_cost_usd"] > 0
    assert b["human"]["total_cost_usd"] >= b["human"]["labour_cost_usd"]


def test_setup_can_be_excluded():
    b = compute_breakdown(n_responses=210, human_hr_rate=20.0, include_setup=False)
    assert b["human"]["setup_cost_usd"] == 0


def test_adjudication_omitted_without_rate():
    """Without a disagreement rate, adjudication is OMITTED (not silently 0)."""
    b = compute_breakdown(n_responses=210)
    assert b["MODEL"]["adjudication"].startswith("OMITTED")
    assert b["human"]["adjudication_cost_usd"] == 0


def test_adjudication_counted_when_rate_provided():
    """With a disagreement rate, adjudication cost is a MODEL estimate > 0."""
    b = compute_breakdown(
        n_responses=210,
        disagreement_rate=0.10,
        adjudication_seconds_per_case=300.0,
        human_hr_rate=20.0,
    )
    assert b["MODEL"]["adjudication"].startswith("MODEL")
    assert b["human"]["adjudication_cost_usd"] > 0


def test_nominal_ratio_has_caveat():
    """The ratio must always carry the apples-to-oranges caveat."""
    b = compute_breakdown(n_responses=210)
    assert "apples-to-oranges" in b["comparison"]["caveat"].lower()
    assert "not a single universally-valid" in b["comparison"]["caveat"].lower()


def test_auto_uses_zero_labour():
    b = compute_breakdown(n_responses=210)
    assert b["auto"]["labour_hours"] == 0.0
    assert b["comparison"]["auto_labour_hours"] == 0.0


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        compute_breakdown(n_responses=0)
