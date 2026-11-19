"""Sealing a split is not the same as having a result.

A re-seal produces templates, a manifest and a passing seal-integrity check, so
it *looks* done. But with no annotations there is no gold and no kappa, and
validating it would let an un-annotated split masquerade as a published
experiment. The gate must refuse it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_rerun_pipeline import _check_experiment_has_annotations  # noqa: E402


class _Args:
    def __init__(self, experiment_id: str, annotations_dir: Path) -> None:
        self.experiment_id = experiment_id
        self.annotations_dir = annotations_dir


def _rater_file(directory: Path, exp_id: str, rater: str, split: str, body: str) -> None:
    (directory / f"{exp_id}_{rater}_{split}.jsonl").write_text(body)


def _write_complete_set(directory: Path, exp_id: str) -> None:
    for split in ("calibration", "heldout"):
        for rater in ("RATER_A", "RATER_B"):
            _rater_file(directory, exp_id, rater, split, '{"internal_key": "K_0001"}\n')


def test_sealed_but_unannotated_experiment_is_rejected(tmp_path):
    """The whole point: nothing judged -> nothing to validate."""
    with pytest.raises(SystemExit):
        _check_experiment_has_annotations(_Args("fresh-seal", tmp_path))


def test_missing_one_rater_file_is_rejected(tmp_path):
    exp_id = "half-annotated"
    _write_complete_set(tmp_path, exp_id)
    (tmp_path / f"{exp_id}_RATER_B_heldout.jsonl").unlink()

    with pytest.raises(SystemExit):
        _check_experiment_has_annotations(_Args(exp_id, tmp_path))


def test_empty_annotation_file_is_rejected(tmp_path):
    """A present-but-empty file is a silent no-op, so it must not pass either."""
    exp_id = "empty-heldout"
    _write_complete_set(tmp_path, exp_id)
    _rater_file(tmp_path, exp_id, "RATER_A", "heldout", "\n  \n")

    with pytest.raises(SystemExit):
        _check_experiment_has_annotations(_Args(exp_id, tmp_path))


def test_complete_annotation_set_passes(tmp_path):
    exp_id = "fully-annotated"
    _write_complete_set(tmp_path, exp_id)
    _check_experiment_has_annotations(_Args(exp_id, tmp_path))  # must not raise
