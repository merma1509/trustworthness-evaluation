"""Abstention is a valid annotation, not a gap.

Rubric 3 ("no skip / no exclusion") and RATER_INSTRUCTIONS 5 both require a
rater who cannot decide to set ``flags.rater_uncertainty = "uncertain"`` and
leave the label blank. That record is *uncertainty data*: it stays in the
dataset and is never forced into an arbitrary label.

The bug these tests pin down: ``ingest_annotations.py`` treated ANY blank label
as a schema violation ("no label assigned (must pick exactly one)"), so a
protocol-perfect file containing one honest abstention failed ingestion.
``experiment_status.py`` made the complementary mistake, counting abstentions as
unfilled records and keeping the experiment PENDING forever.

Both directions are locked here:

1. A declared abstention VALIDATES (and is reported as ``abstained``).
2. A blank label WITHOUT the uncertain flag is still INVALID (real omission).
3. ``uncertain`` always vetoes a vote, so the A≠B disagreement set matches
   ``resolve_disagreements.py``.
4. The status report counts abstentions as filled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_status import build_status, inspect_file
from scripts.ingest_annotations import (
    cross_check_adjudicator,
    validate_annotation_record,
)

_LABEL_FIELD = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}


def _rec(dim: str, label: str = "", uncertain: str = "confident",
         internal_key: str = "K1") -> dict:
    """Build a minimal sealed-style annotation record."""
    ann = {v: "" for v in _LABEL_FIELD.values()}
    ann[_LABEL_FIELD[dim]] = label
    ann["label_source"] = "rater_judgment"
    return {
        "internal_key": internal_key,
        "prompt_record": {"dimension": dim, "prompt_text": "q"},
        "annotations": ann,
        "flags": {
            "rater_uncertainty": uncertain,
            "ai_assistance_used": False,
            "response_invalid": False,
        },
        "metadata": {"notes": ""},
    }


def _write(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


# ── 1. a declared abstention is valid ─────────────────────────────────────
def test_abstention_validates_and_is_flagged():
    res = validate_annotation_record(
        _rec("truthfulness", label="", uncertain="uncertain")
    )

    assert res["valid"] is True
    assert res["problems"] == []
    assert res["abstained"] is True
    assert res["uncertain"] is True
    assert res["label"] == ""


@pytest.mark.parametrize("dim,label", [
    ("safety", "REFUSE"),
    ("truthfulness", "CORRECT_UNCERTAINTY"),
    ("consistency", "CONSISTENT"),
])
def test_labelled_confident_record_is_valid(dim, label):
    res = validate_annotation_record(_rec(dim, label=label))

    assert res["valid"] is True
    assert res["abstained"] is False


# ── 2. a blank label with no uncertain flag is still an omission ──────────
def test_blank_label_without_uncertain_flag_is_invalid():
    res = validate_annotation_record(
        _rec("safety", label="", uncertain="confident")
    )

    assert res["valid"] is False
    assert res["abstained"] is False
    assert any("no label assigned" in p for p in res["problems"])


def test_guess_flag_without_label_is_invalid():
    """``guess`` is not an abstention — it still requires a real label."""
    res = validate_annotation_record(_rec("safety", label="", uncertain="guess"))

    assert res["valid"] is False
    assert res["abstained"] is False


# ── 3. abstention vetoes the vote (matches resolve_disagreements) ─────────
def test_uncertain_with_label_still_counts_as_no_vote(tmp_path):
    """An uncertain rater never creates an A≠B dispute, even if it typed a label."""
    a = _write(tmp_path / "a.jsonl",
               [_rec("safety", label="REFUSE", uncertain="uncertain")])
    b = _write(tmp_path / "b.jsonl",
               [_rec("safety", label="COMPLY")])
    adj = _write(tmp_path / "adj.jsonl", [])

    # An empty adjudicator file is correct here: no valid disagreement exists.
    cross_check_adjudicator(adj, a, b, "heldout")


def test_missing_dispute_in_adjudicator_file_fails(tmp_path):
    a = _write(tmp_path / "a.jsonl", [_rec("safety", label="REFUSE")])
    b = _write(tmp_path / "b.jsonl", [_rec("safety", label="COMPLY")])
    adj = _write(tmp_path / "adj.jsonl", [])

    # The real disagreement is MISSING from the adjudicator file -> must fail.
    with pytest.raises(ValueError):
        cross_check_adjudicator(adj, a, b, "heldout")


def test_adjudicator_file_with_correct_dispute_passes(tmp_path):
    a = _write(tmp_path / "a.jsonl", [_rec("safety", label="REFUSE")])
    b = _write(tmp_path / "b.jsonl", [_rec("safety", label="COMPLY")])
    adj = _write(tmp_path / "adj.jsonl", [_rec("safety", label="REFUSE")])

    cross_check_adjudicator(adj, a, b, "heldout")  # must not raise


# ── 4. the status report counts abstentions as filled ────────────────────
def test_status_counts_abstention_as_filled(tmp_path):
    path = _write(tmp_path / "exp1_RATER_A_calibration.jsonl", [
        _rec("safety", label="REFUSE", internal_key="K1"),
        _rec("safety", label="", uncertain="uncertain", internal_key="K2"),
    ])

    rep = inspect_file(path)

    assert rep["n_records"] == 2
    assert rep["n_abstained"] == 1
    assert rep["n_missing"] == 0        # abstention is NOT a gap
    assert rep["n_invalid"] == 0
    assert rep["n_filled"] == 2


def test_status_reaches_ready_with_abstentions(tmp_path):
    exp = "exp-abstain"
    recs = [_rec("safety", label="REFUSE", internal_key="K1"),
            _rec("safety", label="", uncertain="uncertain", internal_key="K2")]
    for split in ("calibration", "heldout"):
        _write(tmp_path / f"{exp}_RATER_A_{split}.jsonl", recs)
        _write(tmp_path / f"{exp}_RATER_B_{split}.jsonl", recs)
        # Adjudicator sees nothing: no valid disagreement exists.
        _write(tmp_path / f"{exp}_ADJUDICATOR_{split}.jsonl", [])

    st = build_status(tmp_path, exp)

    assert st["status"] == "READY"
    assert all(f.get("n_missing", 0) == 0 for f in st["per_file"])


def test_uncertain_never_creates_a_disagreement_in_status(tmp_path):
    exp = "exp-u"
    recs_a = [_rec("safety", label="REFUSE", internal_key="K1")]
    recs_b = [_rec("safety", label="COMPLY", uncertain="uncertain",
                   internal_key="K1")]
    for split in ("calibration", "heldout"):
        _write(tmp_path / f"{exp}_RATER_A_{split}.jsonl", recs_a)
        _write(tmp_path / f"{exp}_RATER_B_{split}.jsonl", recs_b)

    st = build_status(tmp_path, exp)

    # A voted, B abstained -> no valid pair -> no dispute for the adjudicator.
    assert st["heldout_agreements"]["n_disagreements_A_vs_B"] == 0
