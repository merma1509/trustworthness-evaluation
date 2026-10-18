"""Tests for the disagreements-only Adjudicator handoff.

``scripts/prepare_adjudicator.py`` is the step that constructs the ADJUDICATOR
file from ONLY the records where RATER_A and RATER_B disagree, rebuilt from the
*sealed* template so the adjudicator stays double-blind (never sees A's or B's
label).

These tests protect three properties:

1. **Disagreements-only** — records where A == B are NOT sent to the
   adjudicator; records where A != B (both valid votes) ARE.
2. **Double-blind** — the produced file carries no A/B label and no
   ground-truth/identity field; it is rebuilt from the sealed template.
3. **No silent dropping** — a disagreement key whose sealed template is missing
   raises (fails closed) instead of producing a silently-incomplete file.
"""

import json

import pytest

from scripts.prepare_adjudicator import (
    _disagreement_keys,
    _strip_annotation_fields,
    prepare_adjudicator,
)


# ── helpers ────────────────────────────────────────────────────────────────
def _ann(internal_key, dim, label, uncertain=False):
    """One rater annotation record in the sealed-template schema."""
    return {
        "internal_key": internal_key,
        "prompt_record": {
            "dimension": dim,
            "prompt_id": "",
            "prompt_text": "prompt-text",
            "model_response": "model-response",
        },
        "annotations": {
            "safety_label": label if dim == "safety" else "",
            "truthfulness_label": label if dim == "truthfulness" else "",
            "consistency_label": label if dim == "consistency" else "",
            "label_source": "rater_judgment",
        },
        "flags": {
            "rater_uncertainty": "uncertain" if uncertain else "",
            "ai_assistance_used": False,
            "ai_assistance_description": "",
            "response_invalid": False,
        },
        "metadata": {"notes": "note", "declared_conflicts": []},
    }


def _sealed(internal_key, dim, prompt="prompt-text", response="model-response"):
    """A full sealed template record (what a rater SHOULD only ever see)."""
    # Deliberately include a sensitive field name with a non-empty value to
    # assert the double-blind rebuild strips it even if the template regressed.
    return {
        "internal_key": internal_key,
        "prompt_record": {
            "dimension": dim,
            "prompt_id": "",
            "prompt_text": prompt,
            "model_response": response,
            "model_id": "gemma3:4b",  # must be stripped by _strip_annotation_fields
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
        "metadata": {"notes": "note", "declared_conflicts": []},
    }


def _write_jsonl(path, recs):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
        encoding="utf-8",
    )


def _has_nonempty_field(node, name: str) -> bool:
    """True if ``node`` (dict/list tree) has a NON-EMPTY value under ``name``."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == name and v not in ("", None) and v != [] and v != {}:
                return True
            if _has_nonempty_field(v, name):
                return True
    elif isinstance(node, list):
        return any(_has_nonempty_field(item, name) for item in node)
    return False


# ── disagreements-only selection ───────────────────────────────────────────
def test_disagreement_keys_only_include_a_neq_b():
    """Only keys where both raters cast *different valid* votes are selected."""
    a = {
        ("safety", "K1"): _ann("K1", "safety", "REFUSE"),
        ("safety", "K2"): _ann("K2", "safety", "REFUSE"),
        ("safety", "K3"): _ann("K3", "safety", "REFUSE"),
        ("safety", "K4"): _ann("K4", "safety", "REFUSE"),
    }
    b = {
        ("safety", "K1"): _ann("K1", "safety", "REFUSE"),   # agreement
        ("safety", "K2"): _ann("K2", "safety", "COMPLY"),   # disagreement
        ("safety", "K3"): _ann("K3", "safety", "REFUSE"),   # B uncertain -> no valid vote
        ("safety", "K5"): _ann("K5", "safety", "COMPLY"),   # only in B
    }
    keys = _disagreement_keys(a, b)
    assert keys == [("safety", "K2")]


def test_both_must_be_valid_votes():
    """An uncertain or missing label on either side excludes the key."""
    a = {
        ("safety", "K1"): _ann("K1", "safety", "REFUSE"),
        ("safety", "K2"): _ann("K2", "safety", "REFUSE", uncertain=True),
        ("safety", "K3"): _ann("K3", "safety", "", uncertain=True),
    }
    b = {
        ("safety", "K1"): _ann("K1", "safety", "COMPLY"),
        ("safety", "K2"): _ann("K2", "safety", "COMPLY"),
        ("safety", "K3"): _ann("K3", "safety", "COMPLY"),
    }
    keys = _disagreement_keys(a, b)
    # K1 disagrees but BOTH valid -> selected.
    # K2: A uncertain -> no valid vote -> excluded.
    # K3: A has no label (and uncertain) -> excluded.
    assert keys == [("safety", "K1")]


# ── double-blind rebuild from the sealed template ─────────────────────────
def test_output_contains_only_disagreement_keys(tmp_path):
    """The adjudicator file has exactly the A≠B records, nothing else."""
    sealed_recs = [
        _sealed("K1", "safety", "p1", "r1"),
        _sealed("K2", "safety", "p2", "r2"),
        _sealed("K3", "safety", "p3", "r3"),
    ]
    a_recs = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "REFUSE"),
        _ann("K3", "safety", "REFUSE"),
    ]
    b_recs = [
        _ann("K1", "safety", "REFUSE"),   # agree
        _ann("K2", "safety", "COMPLY"),   # disagree
        _ann("K3", "safety", "REFUSE"),   # agree
    ]

    sealed_f = tmp_path / "sealed.jsonl"
    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    out_f = tmp_path / "adjudicator.jsonl"
    _write_jsonl(sealed_f, sealed_recs)
    _write_jsonl(a_f, a_recs)
    _write_jsonl(b_f, b_recs)

    report = prepare_adjudicator(
        a_f, b_f, sealed_f, out_f,
        experiment_id="exp", split="heldout",
    )

    assert report["n_disagreements_sent_to_adjudicator"] == 1
    assert report["n_records_adjudicator_file"] == 1
    assert out_f.exists()
    out_recs = [json.loads(line) for line in out_f.open() if line.strip()]
    assert len(out_recs) == 1
    assert out_recs[0]["internal_key"] == "K2"
    # content is from the sealed template (prompt + response preserved)
    assert out_recs[0]["prompt_record"]["prompt_text"] == "p2"
    assert out_recs[0]["prompt_record"]["model_response"] == "r2"


def test_output_is_double_blind_no_rater_labels(tmp_path):
    """The produced adjudicator record must not leak A/B labels or ids."""
    sealed_recs = [
        _sealed("K1", "truthfulness", "p1", "r1"),
        _sealed("K2", "truthfulness", "p2", "r2"),
    ]
    a_recs = [
        _ann("K1", "truthfulness", "CORRECT_UNCERTAINTY"),
        _ann("K2", "truthfulness", "CORRECT_UNCERTAINTY"),
    ]
    b_recs = [
        _ann("K1", "truthfulness", "CORRECT_UNCERTAINTY"),
        _ann("K2", "truthfulness", "INCORRECT_FABRICATION"),
    ]
    sealed_f = tmp_path / "sealed.jsonl"
    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    out_f = tmp_path / "adj.jsonl"
    _write_jsonl(sealed_f, sealed_recs)
    _write_jsonl(a_f, a_recs)
    _write_jsonl(b_f, b_recs)

    prepare_adjudicator(
        a_f, b_f, sealed_f, out_f,
        experiment_id="exp", split="calibration",
    )
    out_recs = [json.loads(line) for line in out_f.open() if line.strip()]
    assert len(out_recs) == 1  # only K2
    rec = out_recs[0]

    # No label of ANY kind survives.
    for key in ("safety_label", "truthfulness_label", "consistency_label"):
        assert rec["annotations"][key] == ""
    # model_id and anything identity-like must be blanked even if template had it.
    assert rec["prompt_record"]["model_id"] == ""
    for banned in ("auto_label", "model_id", "attack_type", "prompt_id",
                   "expected_behavior", "score"):
        assert not _has_nonempty_field(rec, banned), f"leaked field: {banned}"


def test_strip_annotation_fields_blanks_everything():
    """The sanitizer blanks flags/metadata too, not just the label fields."""
    rec = _sealed("K1", "safety")
    rec["flags"]["difficult_case"] = True
    rec["flags"]["rater_uncertainty"] = "guess"
    rec["metadata"]["notes"] = "researcher note"
    out = _strip_annotation_fields(rec)
    assert out["flags"]["difficult_case"] is False
    assert out["flags"]["rater_uncertainty"] == ""
    assert out["metadata"]["notes"] == ""
    assert out["annotations"]["safety_label"] in ("", None)


# ── fail-closed coverage ──────────────────────────────────────────────────
def test_missing_sealed_template_fails_closed(tmp_path):
    """A disagreement key missing from the sealed template must raise, not drop."""
    sealed_recs = [_sealed("K1", "safety", "p1", "r1")]
    a_recs = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "REFUSE"),
    ]
    b_recs = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "COMPLY"),  # disagreement on a key NOT in template
    ]
    sealed_f = tmp_path / "sealed.jsonl"
    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    out_f = tmp_path / "adj.jsonl"
    _write_jsonl(sealed_f, sealed_recs)
    _write_jsonl(a_f, a_recs)
    _write_jsonl(b_f, b_recs)

    with pytest.raises(RuntimeError, match="missing these disagreement keys"):
        prepare_adjudicator(
            a_f, b_f, sealed_f, out_f,
            experiment_id="exp", split="heldout",
        )
    # Fails closed: nothing written.
    assert not out_f.exists()


# ── fresh-rebuild (delete-then-write) ─────────────────────────────────────
def test_prepare_replaces_existing_adjudicator_file(tmp_path):
    """``prepare_adjudicator`` must REPLACE (not append to / not keep) any
    pre-existing ADJUDICATOR file.

    This matters for the real flow: ``make experiment-annotate RATERS=ADJUDICATOR``
    first leaves a FULL-SIZE template (147/63 records) at the same path, and the
    handoff script must wipe it so the Rater Annotation tab serves the
    adjudicator the disagreements-only file — never a stale full-size one.
    """
    sealed_recs = [
        _sealed("K1", "safety", "p1", "r1"),
        _sealed("K2", "safety", "p2", "r2"),
    ]
    a_recs = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "REFUSE"),
    ]
    b_recs = [
        _ann("K1", "safety", "REFUSE"),   # agree
        _ann("K2", "safety", "COMPLY"),   # disagree
    ]
    sealed_f = tmp_path / "sealed.jsonl"
    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    out_f = tmp_path / "adjudicator.jsonl"

    _write_jsonl(sealed_f, sealed_recs)
    _write_jsonl(a_f, a_recs)
    _write_jsonl(b_f, b_recs)

    # Simulate the stale FULL-SIZE template left by `experiment-annotate`.
    _write_jsonl(out_f, [{"internal_key": "STALE", "junk": True}] * 100)

    report = prepare_adjudicator(
        a_f, b_f, sealed_f, out_f,
        experiment_id="exp", split="heldout",
    )

    assert report["n_disagreements_sent_to_adjudicator"] == 1
    assert out_f.exists()
    out_recs = [json.loads(line) for line in out_f.open() if line.strip()]
    # ONLY the fresh disagreements-only records survive — no stale tail.
    assert len(out_recs) == 1
    assert out_recs[0]["internal_key"] == "K2"
    assert out_recs[0]["prompt_record"]["prompt_text"] == "p2"
