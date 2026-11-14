"""``--inter-rater-calibration`` must do something, and must not leak into held-out.

Bug this pins down: the flag was declared on the argument parser and read by
nothing. Passing it silently produced identical output, so a caller could
believe calibration agreement had been measured when it had not — a dead
parameter is a claim that fails silently.

Two invariants:

1. supplying the flag adds a separate ``inter_rater_calibration_A_vs_B`` block;
2. the held-out ``inter_rater_A_vs_B`` block is byte-identical either way, so
   calibration records can never dilute the reliability figure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "trustworthiness-validation-2026-10-16-042935"
ANN = ROOT / "experiment" / "annotations"
GOLD = ROOT / "experiment" / "gold" / "gold_labels.jsonl"
AUTO = ROOT / "experiment" / "sealed" / "labels" / "auto_labels.jsonl"
RESOLUTIONS = ROOT / "experiment" / "agreements" / f"{EXP_ID}_heldout_disagreements.json"


def _run(tmp_path: Path, calibration: bool) -> dict:
    cmd = [
        sys.executable,
        "scripts/report_agreement.py",
        "--gold", str(GOLD),
        "--auto-labels", str(AUTO),
        "--rater-a", str(ANN / f"{EXP_ID}_RATER_A_heldout.jsonl"),
        str(ANN / f"{EXP_ID}_RATER_A_calibration.jsonl"),
        "--rater-b", str(ANN / f"{EXP_ID}_RATER_B_heldout.jsonl"),
        str(ANN / f"{EXP_ID}_RATER_B_calibration.jsonl"),
        "--out", str(tmp_path / "agreement.json"),
        "--resolutions", str(RESOLUTIONS),
    ]
    if calibration:
        cmd += [
            "--inter-rater-calibration",
            str(ANN / f"{EXP_ID}_RATER_A_calibration.jsonl"),
            str(ANN / f"{EXP_ID}_RATER_B_calibration.jsonl"),
        ]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads((tmp_path / "agreement.json").read_text())


@pytest.fixture(scope="module")
def without_flag(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("nocal"), calibration=False)


@pytest.fixture(scope="module")
def with_flag(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("cal"), calibration=True)


def test_flag_adds_calibration_block(without_flag, with_flag) -> None:
    """The flag must change the output; otherwise it is dead again."""
    assert without_flag.get("inter_rater_calibration_A_vs_B") in (None, {}) or (
        without_flag["inter_rater_calibration_A_vs_B"].get("available") is False
    ), "omitting the flag must not fabricate a measurement"

    block = with_flag.get("inter_rater_calibration_A_vs_B") or {}
    assert block.get("available") is not False, "flag supplied but block unavailable"
    assert block, "flag supplied but no calibration block emitted"
    assert "safety" in block, f"expected per-dimension entries, got {sorted(block)}"


def test_calibration_does_not_leak_into_heldout_block(without_flag, with_flag) -> None:
    """The held-out reliability figure must be identical with and without it."""
    assert with_flag["inter_rater_A_vs_B"] == without_flag["inter_rater_A_vs_B"]

    held = with_flag["inter_rater_A_vs_B"]["safety"]["n"]
    calib = with_flag["inter_rater_calibration_A_vs_B"]["safety"]["n"]
    assert calib > held, "calibration split should be larger; otherwise no leak proof"


def test_missing_calibration_file_reports_unavailable(tmp_path) -> None:
    """A bad path must say 'not measured', never an empty-but-present result."""
    cmd = [
        sys.executable,
        "scripts/report_agreement.py",
        "--gold", str(GOLD),
        "--auto-labels", str(AUTO),
        "--rater-a", str(ANN / f"{EXP_ID}_RATER_A_heldout.jsonl"),
        "--rater-b", str(ANN / f"{EXP_ID}_RATER_B_heldout.jsonl"),
        "--out", str(tmp_path / "a.json"),
        "--resolutions", str(RESOLUTIONS),
        "--inter-rater-calibration",
        "does_not_exist_RATER_A.jsonl",
        "does_not_exist_RATER_B.jsonl",
    ]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    block = json.loads((tmp_path / "a.json").read_text())[
        "inter_rater_calibration_A_vs_B"
    ]
    assert block["available"] is False
    assert block["reason"]
