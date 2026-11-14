"""The split-leak check must compare *stimuli*, not whole group sets.

Bug this pins down: ``diagnose_textual_overlap()`` indexed calibration
consistency records by ``(model, frozenset(member_texts))`` — the whole group as
one atomic key — and compared held-out groups against that key. Re-bucketing
changes group composition, so a leaked stimulus was invisible whenever the two
groups differed in size (3-member held-out group vs 2-member calibration group
sharing one prompt). The check therefore reported ``n_family_overlap_pairs = 0``
while 8 of 22 held-out consistency records genuinely repeated a calibration
(model, prompt) pair — a false negative on the single most important design
invariant.

The fix evaluates the family key **per group member**. These tests prove the
check now fails closed on a partially shared group.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_design import diagnose_textual_overlap

_GEMMA = "gemma3:4b"
_LLAMA = "llama3.1:8b"


def _consistency_record(internal_key: str, prompts: list[str], response: str) -> dict:
    """A sealed consistency template record with the given member prompts."""
    return {
        "internal_key": internal_key,
        "prompt_record": {"dimension": "consistency"},
        "pairs": [{"prompt": p, "response": response} for p in prompts],
    }


def _write_templates(
    root: Path,
    experiment_id: str,
    calibration: list[dict],
    heldout: list[dict],
) -> tuple[Path, Path]:
    """Write both splits and return their (calibration, heldout) directories."""
    cal_dir = root / "calibration"
    hold_dir = root / "heldout"
    for directory, split, records in (
        (cal_dir, "calibration", calibration),
        (hold_dir, "heldout", heldout),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{experiment_id}_{split}_template.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return cal_dir, hold_dir


def _write_auto_labels(labels_path: Path, model_of: dict[str, str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"internal_key": key, "model_id": model}
        for key, model in model_of.items()
    ]
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_partially_shared_group_is_detected(tmp_path: Path) -> None:
    """A held-out group sharing ONE stimulus with calibration must be flagged.

    This is the exact shape that produced the false zero: the calibration group
    has 2 members, the held-out group has 3, and they overlap on one prompt.
    Whole-set comparison sees two different sets and reports no overlap.
    """
    experiment_id = "exp-test"
    templates_dir = tmp_path / "templates"
    labels_path = tmp_path / "labels" / "auto_labels.jsonl"

    calibration = [
        _consistency_record("K_0001", ["what is 2 + 2?", "what is two plus two?"], "4"),
    ]
    heldout = [
        _consistency_record(
            "K_0002",
            ["what is 2 + 2?", "what is 2+2?", "two plus two is?"],
            "4",
        ),
    ]
    cal_dir, hold_dir = _write_templates(
        templates_dir, experiment_id, calibration, heldout
    )
    _write_auto_labels(labels_path, {"K_0001": _GEMMA, "K_0002": _GEMMA})

    res = diagnose_textual_overlap(cal_dir, hold_dir, labels_path)

    assert res["n_family_overlap_pairs"] >= 1, (
        "a shared stimulus inside differently-sized groups must be a blocker; "
        "the whole-set comparison missed exactly this case"
    )
    # The whole-set overlap is genuinely absent, which is why it cannot be used
    # as the blocking check.
    assert res["n_exact_group_overlap_pairs"] == 0
    assert res["ok"] is False


def test_disjoint_groups_pass(tmp_path: Path) -> None:
    """Groups with no shared stimulus must not be flagged."""
    experiment_id = "exp-test"
    templates_dir = tmp_path / "templates"
    labels_path = tmp_path / "labels" / "auto_labels.jsonl"

    calibration = [
        _consistency_record("K_0001", ["what is 2 + 2?", "what is two plus two?"], "4"),
    ]
    heldout = [
        _consistency_record("K_0002", ["name a prime number", "list a prime"], "7"),
    ]
    cal_dir, hold_dir = _write_templates(
        templates_dir, experiment_id, calibration, heldout
    )
    _write_auto_labels(labels_path, {"K_0001": _GEMMA, "K_0002": _GEMMA})

    res = diagnose_textual_overlap(cal_dir, hold_dir, labels_path)

    assert res["n_family_overlap_pairs"] == 0
    assert res["ok"] is True


def test_same_text_different_model_is_not_a_family_blocker(tmp_path: Path) -> None:
    """Identical text under a DIFFERENT model is a warning, not a blocker.

    A content-aware split keys on (model, text); the same prompt asked of two
    models is legitimately split across calibration and held-out.
    """
    experiment_id = "exp-test"
    templates_dir = tmp_path / "templates"
    labels_path = tmp_path / "labels" / "auto_labels.jsonl"

    calibration = [
        _consistency_record("K_0001", ["what is 2 + 2?", "what is two plus two?"], "4"),
    ]
    heldout = [
        _consistency_record("K_0002", ["what is 2 + 2?", "what is two plus two?"], "4"),
    ]
    cal_dir, hold_dir = _write_templates(
        templates_dir, experiment_id, calibration, heldout
    )
    _write_auto_labels(labels_path, {"K_0001": _GEMMA, "K_0002": _LLAMA})

    res = diagnose_textual_overlap(cal_dir, hold_dir, labels_path)

    assert res["n_family_overlap_pairs"] == 0
    assert res["n_cross_model_text_overlap_pairs"] >= 1
    assert res["ok"] is True


