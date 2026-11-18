"""Every recorded key must be accounted for — in EVERY exclusion bucket.

Two bugs are pinned here, both of the same shape: a coverage invariant that
silently disagreed with the data it was guarding.

1. **The gate only counted one exclusion reason.** ``_check_coverage`` summed
   ``n_scored + n_unresolvable + n_excluded_response_invalid`` and compared to
   ``n_recorded``. A consistency display with a single response is excluded for
   a *different* reason (``unscorable_consistency``), so the real experiment
   (63 recorded = 49 scored + 11 unresolvable + 3 unscorable) failed the check
   and looked like data had been dropped. Nothing had been.

2. **``report_agreement`` dropped the very counter the gate needed.** The
   report's ``resolution_coverage`` block carried ``n_excluded_response_invalid``
   but not ``n_excluded_unscorable_consistency``, so the same false failure
   reproduced even after the gate was fixed.

3. **The validator resolved without the template.** Display sizes live only in
   the sealed template, so a run without ``--templates`` scored 51 instead of
   the published 49 — the gate validated numbers nobody reports.

The shared rule now lives in ``src.labels`` (``coverage_gaps``), so the gate,
the report and the driver cannot drift apart again.
"""

from __future__ import annotations

import json

import pytest

from scripts.report_agreement import _load_resolution_coverage
from scripts.validate_rerun_pipeline import _check_coverage
from src.labels import accounted_records, coverage_gaps

# A report shaped exactly like the real held-out one: the invariant only holds
# once the singleton-consistency bucket is included.
_REAL_SHAPE = {
    "n_recorded": 63,
    "n_scored": 49,
    "n_unresolvable": 11,
    "n_excluded_response_invalid": 0,
    "n_excluded_unscorable_consistency": 3,
    "exclusion_reason_counts": {
        "singleton_consistency_display": 3,
    },
}


# ── the shared rule ──────────────────────────────────────────────────────────
def test_accounted_records_sums_every_exclusion_bucket():
    assert accounted_records(_REAL_SHAPE) == 63


def test_missing_optional_counter_is_zero_not_unknown():
    """A run that never excluded anything legitimately omits the counter."""
    report = {
        "n_recorded": 10,
        "n_scored": 8,
        "n_unresolvable": 2,
        "n_excluded_response_invalid": 0,
    }
    assert accounted_records(report) == 10
    assert coverage_gaps(report) == []


def test_missing_mandatory_counter_is_reported_as_unknown():
    assert accounted_records({"n_recorded": 5}) is None
    assert coverage_gaps({"n_recorded": 5})


def test_real_shape_has_no_gaps():
    assert coverage_gaps(_REAL_SHAPE) == []


def test_ignoring_singletons_is_detected_as_a_gap():
    """The exact bug: scored + unresolvable + response_invalid alone != recorded."""
    buggy = {k: v for k, v in _REAL_SHAPE.items()
             if k != "n_excluded_unscorable_consistency"}
    gaps = coverage_gaps(buggy)
    assert gaps, "dropping the singleton bucket must be flagged"
    assert "60" in gaps[0] and "63" in gaps[0]


def test_reason_counts_must_match_the_counters():
    inconsistent = dict(_REAL_SHAPE)
    inconsistent["exclusion_reason_counts"] = {"singleton_consistency_display": 1}
    gaps = coverage_gaps(inconsistent)
    assert any("exclusion_reason_counts" in g for g in gaps)


# ── the gate now accepts a correct report and still fails a broken one ───────
def test_gate_passes_on_the_real_shape():
    _check_coverage(_REAL_SHAPE)  # must not raise


def test_gate_still_fails_when_a_record_is_unaccounted_for():
    broken = dict(_REAL_SHAPE)
    broken["n_scored"] = 48  # one record vanished
    with pytest.raises(SystemExit):
        _check_coverage(broken)


def test_gate_fails_closed_on_a_zero_scored_report():
    """0 scored with records present means blank templates, never a valid run."""
    with pytest.raises(SystemExit):
        _check_coverage({**_REAL_SHAPE, "n_scored": 0})


# ── the report must carry the counter the gate reads ────────────────────────
def test_report_exposes_the_singleton_bucket(tmp_path):
    src = tmp_path / "resolutions.json"
    src.write_text(json.dumps(_REAL_SHAPE))

    block = _load_resolution_coverage(src)

    assert block["n_excluded_unscorable_consistency"] == 3, (
        "report_agreement must expose the singleton bucket; omitting it made "
        "the coverage block fail its own invariant"
    )
    assert coverage_gaps(block) == []
