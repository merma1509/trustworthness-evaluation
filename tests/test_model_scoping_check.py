"""The model-scoping check must be evidence-based, not a tautology.

Bug this pins down: ``diagnose_model_scoping()`` computed ``n_mixed`` but never
incremented it, so the "display mixes two models" check always reported ``0`` —
a silently skipped validation that made the whole consistency dimension look
clean no matter what the sealer produced.

The fix resolves each anonymous ``pairs[]`` entry back to its source model via
the raw outputs and compares that against ``metadata.display_model``. These
tests prove the check FAILS CLOSED on real defects, and only passes a display
that is genuinely single-model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose_design import diagnose_model_scoping

_GEMMA = "gemma3:4b"
_LLAMA = "llama3.1:8b"


def _raw_row(prompt: str, response: str, model: str) -> dict:
    return {"prompt_text": prompt, "response": response, "model_id": model}


def _write_raw(raw_dir: Path, rows) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "pool_consistency.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _display(internal_key: str, pairs, display_model: str = _GEMMA) -> dict:
    return {
        "internal_key": internal_key,
        "prompt_record": {"dimension": "consistency"},
        "metadata": {"display_model": display_model},
        "pairs": [{"prompt": p, "response": r} for p, r in pairs],
    }


def _write_templates(tpl_dir: Path, records, split: str = "calibration") -> Path:
    tpl_dir.mkdir(parents=True, exist_ok=True)
    path = tpl_dir / f"exp_{split}_template.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture
def pool(tmp_path):
    """Raw pool: three gemma responses and one llama response."""
    raw = tmp_path / "raw"
    _write_raw(raw, [
        _raw_row("Q1", "A1", _GEMMA),
        _raw_row("Q2", "A2", _GEMMA),
        _raw_row("Q3", "A3", _GEMMA),
        _raw_row("Q4", "A4", _LLAMA),
    ])
    return raw


# ── the old bug: always zero ──────────────────────────────────────────────
def test_mixed_display_is_detected(tmp_path, pool):
    """A display whose pairs come from two models must be a blocker."""
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [
        # display_model claims gemma, but one pair was produced by llama
        _display("K1", [("Q1", "A1"), ("Q2", "A2"), ("Q4", "A4")], _GEMMA),
    ])

    res = diagnose_model_scoping(tpl, pool)

    assert res["ok"] is False
    assert res["n_displays_mixing_models"] == 1
    assert res["mixed_examples"][0]["resolved_models"] == [_GEMMA, _LLAMA]


def test_single_model_display_passes(tmp_path, pool):
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [
        _display("K1", [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")], _GEMMA),
    ])

    res = diagnose_model_scoping(tpl, pool)

    assert res["ok"] is True
    assert res["n_displays_mixing_models"] == 0
    assert res["n_unresolved_pairs"] == 0


# ── metadata must agree with the evidence ────────────────────────────────
def test_display_model_mismatch_is_detected(tmp_path, pool):
    """Single-model display, but display_model names the WRONG model."""
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [
        _display("K1", [("Q4", "A4")], _GEMMA),  # actually llama
    ])

    res = diagnose_model_scoping(tpl, pool)

    assert res["ok"] is False
    assert res["n_displays_display_model_mismatch"] == 1
    assert res["mismatch_examples"][0]["resolved_models"] == [_LLAMA]


def test_unscoped_display_without_display_model_fails(tmp_path, pool):
    tpl = tmp_path / "tpl"
    rec = _display("K1", [("Q1", "A1")])
    rec["metadata"] = {}  # no display_model at all
    _write_templates(tpl, [rec])

    res = diagnose_model_scoping(tpl, pool)

    assert res["ok"] is False
    assert res["n_displays_unscoped"] == 1


# ── fail closed when the evidence is missing ─────────────────────────────
def test_untraceable_pair_fails_closed(tmp_path, pool):
    """A pair absent from the raw pool cannot be proven single-model."""
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [
        _display("K1", [("Q1", "A1"), ("Q999", "GHOST")], _GEMMA),
    ])

    res = diagnose_model_scoping(tpl, pool)

    assert res["ok"] is False
    assert res["n_unresolved_pairs"] == 1


def test_no_raw_dir_reports_evidence_unavailable(tmp_path):
    """Without the raw pool the evidence checks are skipped — and said so."""
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [_display("K1", [("Q1", "A1")], _GEMMA)])

    res = diagnose_model_scoping(tpl, None)

    # Metadata-only checks still run, so a scoped display is not a blocker...
    assert res["evidence_available"] is False
    assert res["n_displays_mixing_models"] == 0
    # ...but the caller must be able to tell this apart from a real proof.
    assert res["n_unresolved_pairs"] == 0


def test_missing_raw_dir_path_is_treated_as_unavailable(tmp_path):
    tpl = tmp_path / "tpl"
    _write_templates(tpl, [_display("K1", [("Q1", "A1")], _GEMMA)])

    res = diagnose_model_scoping(tpl, tmp_path / "does_not_exist")

    assert res["evidence_available"] is False


def test_non_consistency_records_are_ignored(tmp_path, pool):
    tpl = tmp_path / "tpl"
    rec = _display("K1", [("Q1", "A1")])
    rec["prompt_record"]["dimension"] = "safety"
    _write_templates(tpl, [rec])

    res = diagnose_model_scoping(tpl, pool)

    assert res["n_consistency_displays"] == 0
    assert res["ok"] is True
