"""Reproducibility regression tests for the blinded-annotation generator.

Background
----------
``generate_blinded_annotation.py`` builds a calibration / held-out split by
shuffling the sorted list of *units* with a seeded ``random.Random`` and then
slicing it. The original implementation converted the two slices into Python
``set`` objects (``set(units[:n_cal])`` / ``set(units[n_cal:])``) and iterated
them while emitting output rows.

Python ``set`` iteration order is *not guaranteed*: it depends on the process's
hash randomisation seed (``PYTHONHASHSEED``), so it differs run-to-run. Two
consequences:

* the *order* of output rows changes on every regeneration; and
* the sequential anonymised ids (``U_0001``, ``U_0002``, ...) get assigned to
  *different* physical units each run, so a regenerated split no longer matches
  already-collected human annotations nor the analyst's ground-truth mapping.

These tests guard against regressions back to that non-deterministic behaviour.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "generate_blinded_annotation.py"
AUDIT_PATH = PROJECT_ROOT / "results" / "audit" / "all_audit.jsonl"
RAW_DIR = PROJECT_ROOT / "results" / "raw_outputs"

SEED = 42
CAL_RATIO = 0.3


def _run_generator(out_dir: Path, tmp_path: Path) -> dict:
    """Run the generator into ``out_dir`` and return a content manifest."""
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--audit", str(AUDIT_PATH),
            "--raw", str(RAW_DIR),
            "--output", str(out_dir),
            "--calibration-ratio", str(CAL_RATIO),
            "--seed", str(SEED),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    manifest = {}
    for rel in (
        "blinded_annotation_calibration.jsonl",
        "blinded_annotation_heldout.jsonl",
        "ground_truth_blinded.json",
        "manifest.json",
    ):
        p = out_dir / rel
        assert p.exists(), f"missing generated file {rel}"
        manifest[rel] = p.read_text(encoding="utf-8")
    return manifest


@pytest.mark.skipif(
    not (SCRIPT.exists() and AUDIT_PATH.exists() and RAW_DIR.exists()),
    reason="blinded generator inputs not present",
)
def test_blinded_generation_is_deterministic(tmp_path):
    """Two runs with identical inputs + seed must produce identical output.

    This is the regression test for the set-iteration bug: iterating a Python
    ``set`` of unit keys reorders the output and reassigns anon_ids non-
    deterministically (due to hash randomisation). The generator must instead
    slice the seeded/shuffled list deterministically.
    """
    out_a = tmp_path / "gen_a"
    out_b = tmp_path / "gen_b"
    manifest_a = _run_generator(out_a, tmp_path)
    manifest_b = _run_generator(out_b, tmp_path)

    for rel in manifest_a:
        assert manifest_a[rel] == manifest_b[rel], (
            f"non-deterministic regeneration: '{rel}' differs between two "
            f"runs with the same seed ({SEED}). Likely a set-iteration bug."
        )


@pytest.mark.skipif(
    not (SCRIPT.exists() and AUDIT_PATH.exists() and RAW_DIR.exists()),
    reason="blinded generator inputs not present",
)
def test_blinded_anon_id_mapping_stable_across_runs(tmp_path):
    """The same physical unit must map to the same anon_id across regenerations.

    Even if rows were re-sorted, re-running must not silently reassign a
    different sequential ``anon_id`` to the same real unit. We verify the
    ``by_anon_id`` ground-truth lookup is byte-identical across two runs.
    """
    out_a = tmp_path / "gen_a"
    out_b = tmp_path / "gen_b"
    gt_a = json.loads(_run_generator(out_a, tmp_path)["ground_truth_blinded.json"])
    gt_b = json.loads(_run_generator(out_b, tmp_path)["ground_truth_blinded.json"])

    assert gt_a["by_anon_id"] == gt_b["by_anon_id"], (
        "anon_id -> real-unit mapping changed across regenerations with the "
        "same seed. The blinded experiment would be impossible to reproduce."
    )
