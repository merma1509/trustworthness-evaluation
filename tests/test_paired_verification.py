"""The paired comparison must be checkable, not merely re-runnable.

``results/paired_comparison.json`` carries the "no significant difference"
claim that the write-up quotes. Before this, the rerun gate never regenerated
it: ``grep paired scripts/validate_rerun_pipeline.py`` was empty, so a shipped
paired figure could drift from the raw outputs and nothing noticed.

Two distinct guarantees are pinned here, because adding the first without the
second is worthless:

1. the artifact is *regenerated* by the gate (so it at least reflects the code
   that is running now);
2. it is *compared against a committed reference* in ``expected_results.json``.

The second point is the trap. ``verify_results`` receives ``--results results``
and the gate has just overwritten ``results/paired_comparison.json``, so
comparing the artifact to itself is a tautology that passes on ANY tampering --
a green check that proves nothing. The test below tampers with a figure and
requires the failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_results import _check_paired_reproducibility

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = json.loads((ROOT / "results" / "expected_results.json").read_text())


def _paired(**overrides) -> dict:
    """A paired report matching the committed reference, with optional edits."""
    dims = json.loads(json.dumps(EXPECTED["paired"]["dimensions"]))
    for dotted, value in overrides.items():
        dim, field = dotted.split("__")
        dims[dim][field] = value
    return {"dimensions": dims}


def test_reference_exists():
    """Without a committed reference the check silently does nothing."""
    assert EXPECTED.get("paired", {}).get("dimensions"), (
        "expected_results.json needs a 'paired' block, otherwise "
        "_check_paired_reproducibility returns early and verifies nothing"
    )


def test_matching_report_passes():
    errors: list[str] = []
    _check_paired_reproducibility(_paired(), EXPECTED, errors)
    assert errors == []


def test_tampered_p_value_is_caught():
    """The exact tautology trap: a changed figure must fail."""
    errors: list[str] = []
    _check_paired_reproducibility(
        _paired(safety__p_value=0.05), EXPECTED, errors
    )
    assert errors, "a tampered p_value must not pass the gate"
    assert "safety.p_value" in errors[0]


def test_tampered_ci_is_caught():
    errors: list[str] = []
    _check_paired_reproducibility(
        _paired(consistency__ci_lower=-0.9), EXPECTED, errors
    )
    assert any("ci_lower" in e for e in errors)


def test_missing_artifact_fails_when_reference_exists():
    """A reference is committed, so a missing artifact is a real failure."""
    errors: list[str] = []
    _check_paired_reproducibility(None, EXPECTED, errors)
    assert errors, "a missing paired report must fail, not be skipped"


def test_no_reference_skips_quietly():
    """Partial result dirs (unit tests, ad-hoc runs) must not be failed.

    Without a committed reference there is nothing to reproduce against, so the
    check is a no-op rather than a spurious error.
    """
    errors: list[str] = []
    _check_paired_reproducibility(None, {}, errors)
    assert errors == []


def test_missing_dimension_fails():
    report = _paired()
    report["dimensions"].pop("safety")
    errors: list[str] = []
    _check_paired_reproducibility(report, EXPECTED, errors)
    assert errors
