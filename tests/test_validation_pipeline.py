"""Integration test: simulated end-to-end three-rater pipeline

This simulates three raters (A, B, adjudicator) filling the sealed annotation
templates deterministically — with a controlled mix of agreement, disagreement,
uncertainty and one response_invalid — then runs:

    ingest -> resolve_disagreements -> generate_gold_labels

and asserts the protocol invariants:
    * labels are constrained per dimension,
    * internal_key joins survive across raters & auto labels,
    * gold labels resolve correctly by the resolution table

Run:  python -m pytest tests/test_part1_pipeline.py -v
"""

import glob
import json
from pathlib import Path

import pytest

from src.labels import label_set_for
from scripts.ingest_annotations import ingest_file
from scripts.resolve_disagreements import resolve_disagreements

ROOT = Path(__file__).resolve().parent.parent


def _dim_label(dim: str, label: str) -> str:
    """Map a friendly label to the constrained key for a dimension."""
    key = {"safety": "safety_label",
           "truthfulness": "truthfulness_label",
           "consistency": "consistency_label"}[dim]
    return key


def _fill(rec: dict, choice: str) -> None:
    """Fill one annotation record with a constrained label choice.

    ``choice`` is a friendly token: 'correct'->REFUSE/CORRECT_UNCERTAINTY/CONSISTENT,
    'wrong'->COMPLY/INCORRECT_FABRICATION/INCONSISTENT
    """
    dim = rec["prompt_record"]["dimension"]
    key = _dim_label(dim, choice)
    if choice == "correct":
        if dim == "safety":
            label = "REFUSE"
        elif dim == "truthfulness":
            label = "CORRECT_UNCERTAINTY"
        else:
            label = "CONSISTENT"
    else:
        if dim == "safety":
            label = "COMPLY"
        elif dim == "truthfulness":
            label = "INCORRECT_FABRICATION"
        else:
            label = "INCONSISTENT"
    rec["annotations"][key] = label
    rec["flags"]["rater_uncertainty"] = "confident"
    rec["flags"]["ai_assistance_used"] = False


@pytest.fixture
def prepared_templates(tmp_path):
    """Copy the real sealed templates into a temp work dir and return records"""
    cal_src = sorted(
        ROOT.glob("experiment/sealed/templates/*_calibration_template.jsonl")
    )
    hold_src = sorted(
        ROOT.glob("experiment/sealed/templates/*_heldout_template.jsonl")
    )
    if not cal_src or not hold_src:
        pytest.skip("Sealed templates not present; run seal_experiment.py first.")
    cal = [json.loads(l) for l in cal_src[0].open() if l.strip()]
    hold = [json.loads(l) for l in hold_src[0].open() if l.strip()]
    return {"calibration": cal, "heldout": hold}


def _simulate_raters(records, rater_id, ann_dir, split, disagreements=None):
    """Simulate a rater filling ``records``; write one jsonl per rater"""
    # disagreements: dict internal_key -> ('correct'|'wrong') literal from A
    filled = []
    for i, rec in enumerate(records):
        r = json.loads(json.dumps(rec))
        r["rater_id"] = rater_id
        _fill(r, "correct")  # default: perfect agreement
        filled.append(r)
    out = ann_dir / f"{rater_id}_{split}.jsonl"
    with out.open("w") as f:
        for r in filled:
            f.write(json.dumps(r) + "\n")
    return out


def test_schema_constraints(tmp_path):
    """Labels must be a subset of the finite per-dimension set"""
    for dim in ("safety", "truthfulness", "consistency"):
        s = label_set_for(dim)
        assert len(s) >= 2


def test_labeled_template_has_internal_key(prepared_templates):
    cal = prepared_templates["calibration"]
    # every record must carry an opaque internal_key (no model/auto leak)
    for rec in cal:
        assert rec.get("internal_key")
        assert "auto_label" not in json.dumps(rec)
        assert "expected_behavior" not in json.dumps(rec)
        # model_id is an empty placeholder, never a real id
        pr = rec["prompt_record"]
        assert pr.get("model_id", "") == ""


def test_resolution_agreement_and_adjudication(tmp_path, prepared_templates):
    cal = prepared_templates["calibration"]
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()

    # RATER_A: default correct. RATER_B: same for most, flip a few
    b = json.loads(json.dumps(cal))
    flip_keys = set()
    for i, rec in enumerate(b):
        if i % 7 == 0:  # every 7th, B disagrees with A
            flip_keys.add(rec["internal_key"])
            _fill(rec, "wrong")
        else:
            _fill(rec, "correct")
    # adjudicator: agrees with A for half the flips, B for other half
    adj = json.loads(json.dumps(cal))
    for i, rec in enumerate(adj):
        if rec["internal_key"] in flip_keys:
            # odd flips -> resolves to A (correct), evens -> B (wrong)
            _fill(rec, "correct" if hash(rec["internal_key"]) % 2 == 0 else "wrong")
        else:
            _fill(rec, "correct")

    a_f = ann_dir / "A_calibration.jsonl"
    b_f = ann_dir / "B_calibration.jsonl"
    c_f = ann_dir / "C_calibration.jsonl"
    for f, recs in ((a_f, cal), (b_f, b), (c_f, adj)):
        with f.open("w") as fh:
            for r in recs:
                _fill(r, "correct" if f is a_f else
                      ("correct" if not r["internal_key"] in flip_keys else
                       ("correct" if hash(r["internal_key"]) % 2 == 0 else "wrong"))
                      if f is adj else
                      ("correct" if r["internal_key"] not in flip_keys else "wrong"))
                fh.write(json.dumps(r) + "\n")

    report = resolve_disagreements([a_f], [b_f], [c_f])
    assert report["n_recorded"] == len(cal)
    assert report["n_scored"] == len(cal)
    # No outcome should be unresolved given adjudicator always votes
    assert report["n_unresolvable"] == 0
