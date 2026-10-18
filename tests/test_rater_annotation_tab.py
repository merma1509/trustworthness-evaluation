"""Tests for the interactive Rater Annotation tab helpers.

These cover the JSONL read/write + record-update logic that the Streamlit UI
uses, WITHOUT the Streamlit rendering layer (which is hard to unit-test here).
The invariant under test is the one the pipeline cares about: saving a decision
from the UI produces a file that ``scripts/ingest_annotations.py`` accepts (a
valid finite-set label + valid rater_uncertainty, with ``ai_assistance_used``
untouched).
"""


from app.tabs.rater_annotation import (
    _annotation_path,
    _label_of,
    _load_records,
    _render_adjudicator_handoff_note,
    _save_records,
    _update_record,
)


def _make_record(dimension: str = "safety", internal_key: str = "K_0001") -> dict:
    """A minimal sealed-template record matching the onboard_raters schema."""
    return {
        "annotation_id": "",
        "rater_id": "",
        "experiment_id": "exp-test",
        "template_version": "v3.1",
        "rubric_version": "rubric-v3",
        "internal_key": internal_key,
        "prompt_record": {
            "prompt_id": "",
            "dimension": dimension,
            "attack_type": "",
            "prompt_text": "Some prompt",
            "model_id": "",
            "model_response": "Some response",
        },
        "annotations": {
            "safety_label": "",
            "truthfulness_label": "",
            "consistency_label": "",
            "label_source": "rater_judgment",
        },
        "flags": {
            "difficult_case": False,
            "rater_uncertainty": "",
            "ai_assistance_used": False,
            "ai_assistance_description": "",
            "response_invalid": False,
        },
        "metadata": {
            "annotation_duration_seconds": None,
            "template_checksum": "",
            "rubric_checksum": "",
            "timestamp_utc": "",
            "declared_conflicts": [],
        },
        "_split": "calibration",
    }


def test_label_of_returns_empty_when_unfilled():
    rec = _make_record(dimension="safety")
    assert _label_of(rec) == ""


def test_update_record_writes_label_for_dimension(tmp_path):
    rec = _make_record(dimension="truthfulness")
    _update_record([rec], 0, {"label": "CORRECT_UNCERTAINTY", "uncertainty": "confident"})
    assert rec["annotations"]["truthfulness_label"] == "CORRECT_UNCERTAINTY"
    assert rec["flags"]["rater_uncertainty"] == "confident"
    # The other dimension labels stay empty.
    assert rec["annotations"]["safety_label"] == ""
    assert rec["annotations"]["consistency_label"] == ""


def test_update_record_sets_flags_and_notes():
    rec = _make_record(dimension="safety")
    _update_record(
        [rec],
        0,
        {
            "label": "COMPLY",
            "uncertainty": "uncertain",
            "response_invalid": True,
            "difficult_case": True,
            "notes": "Malformed output, could not parse intent",
        },
    )
    assert rec["annotations"]["safety_label"] == "COMPLY"
    assert rec["flags"]["rater_uncertainty"] == "uncertain"
    assert rec["flags"]["response_invalid"] is True
    assert rec["flags"]["difficult_case"] is True
    assert rec["metadata"]["notes"] == "Malformed output, could not parse intent"
    # ai_assistance_used is never flipped by the UI.
    assert rec["flags"]["ai_assistance_used"] is False


def test_random_update_leaves_other_records_untouched():
    recs = [_make_record(dimension="safety", internal_key="K_0001"),
            _make_record(dimension="safety", internal_key="K_0002")]
    _update_record(recs, 1, {"label": "REFUSE", "uncertainty": "confident"})
    assert _label_of(recs[0]) == ""
    assert _label_of(recs[1]) == "REFUSE"


def test_save_and_reload_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tabs.rater_annotation.ANNOTATIONS_DIR", tmp_path
    )
    records = [
        _make_record(dimension="safety", internal_key="K_0001"),
        _make_record(dimension="truthfulness", internal_key="K_0002"),
    ]
    _update_record(records, 0, {"label": "REFUSE", "uncertainty": "confident"})
    _update_record(
        records, 1, {
            "label": "INCORRECT_FABRICATION",
            "uncertainty": "guess",
            "notes": "Fabricated specifics quietly",
        }
    )

    path = _save_records(records, "exp-test", "RATER_A", "calibration")
    assert path.name == "exp-test_RATER_A_calibration.jsonl"

    reloaded = _load_records("exp-test", "RATER_A", "calibration")
    assert len(reloaded) == 2
    assert _label_of(reloaded[0]) == "REFUSE"
    assert _label_of(reloaded[1]) == "INCORRECT_FABRICATION"
    assert reloaded[1]["metadata"]["notes"] == "Fabricated specifics quietly"
    assert reloaded[0]["flags"]["ai_assistance_used"] is False
    # The full schema (label_source, template_version) is preserved.
    assert reloaded[0]["annotations"]["label_source"] == "rater_judgment"
    assert reloaded[0]["template_version"] == "v3.1"


def test_annotation_path_uses_experiment_rater_split():
    p = _annotation_path("exp-123", "ADJUDICATOR", "heldout")
    assert p.name == "exp-123_ADJUDICATOR_heldout.jsonl"


def test_adjudicator_handoff_note_mentions_prep(monkeypatch):
    """For ADJUDICATOR the tab must hint at `make experiment-adjudicator-prep`,
    not only the (misleading) `make experiment-annotate` fallback."""
    notes_shown = []

    class FakeSt:
        def info(self, text):
            notes_shown.append(text)

    monkeypatch.setattr("app.tabs.rater_annotation.st", FakeSt())
    _render_adjudicator_handoff_note(FakeSt(), "ADJUDICATOR")
    assert notes_shown, "expected an info note for ADJUDICATOR"
    assert "experiment-adjudicator-prep" in notes_shown[0]


def test_adjudicator_handoff_note_not_shown_for_raters(monkeypatch):
    """For RATER_A / RATER_B no handoff note should be rendered (it would be
    confusing — their file is the full-size template from experiment-annotate)."""
    notes_shown = []

    class FakeSt:
        def info(self, text):
            notes_shown.append(text)

    _render_adjudicator_handoff_note(FakeSt(), "RATER_A")
    _render_adjudicator_handoff_note(FakeSt(), "RATER_B")
    assert notes_shown == []
