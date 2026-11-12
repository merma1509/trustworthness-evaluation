"""Guards that keep a full-size ADJUDICATOR template away from humans.

The bug these tests pin down: ``onboard_raters.py`` used to generate a
FULL-SIZE adjudicator template (147 calibration / 63 heldout records), and
``ingest_annotations.py`` accepted it. The adjudicator is supposed to see ONLY
the records where the independent raters disagree — so a full file both wastes
the adjudicator's effort and (worse) mislabels the protocol.

Three properties are protected:

1. ``onboard_raters`` refuses to emit a full-size ADJUDICATOR template unless
   the caller explicitly opts in (pipeline simulations only).
2. ``onboard_raters`` produces templates for the independent raters by default.
3. ``ingest_annotations.cross_check_adjudicator`` fails closed when the
   ADJUDICATOR file is not exactly the A≠B disagreements subset, and passes when
   it is.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ingest_annotations import (
    _parse_role_split,
    cross_check_adjudicator,
)

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

_DIM_LABEL = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}


# ── helpers ────────────────────────────────────────────────────────────────
def _rec(internal_key: str, dim: str, label: str = "",
         uncertain: bool = False) -> dict:
    """One annotation record in the sealed-template schema."""
    ann = {"label_source": "rater_judgment"}
    if label:
        ann[_DIM_LABEL[dim]] = label
    return {
        "internal_key": internal_key,
        "prompt_record": {
            "dimension": dim,
            "prompt_id": "",
            "prompt_text": "prompt-text",
            "model_response": "model-response",
        },
        "annotations": ann,
        "flags": {"rater_uncertainty": "uncertain" if uncertain else "confident"},
        "metadata": {"notes": ""},
    }


def _write(path: Path, recs: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


# ── 1. onboard_raters role guard ───────────────────────────────────────────
def _sealed_experiment(tmp_path: Path) -> Path:
    """A minimal sealed-experiment dir (manifest + one split template)."""
    exp_dir = tmp_path / "sealed"
    (exp_dir / "templates").mkdir(parents=True)
    (exp_dir / "manifests").mkdir(parents=True)
    (exp_dir / "manifests" / "sealing_manifest.json").write_text(
        json.dumps({"experiment_id": "exp-guard"})
    )
    for split in ("calibration", "heldout"):
        _write(
            exp_dir / "templates" / f"exp-guard_{split}_template.jsonl",
            [_rec("K1", "safety"), _rec("K2", "truthfulness")],
        )
    return exp_dir


def _run_onboard(exp_dir: Path, out_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "scripts/onboard_raters.py",
         "--experiment-dir", str(exp_dir),
         "--out-experiment", str(out_dir),
         *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_onboard_refuses_full_adjudicator_template(tmp_path):
    """Requesting ADJUDICATOR must fail closed — no file, no declaration form."""
    exp_dir = _sealed_experiment(tmp_path)
    out_dir = tmp_path / "out"

    res = _run_onboard(exp_dir, out_dir, "--raters", "RATER_A", "RATER_B", "ADJUDICATOR")

    assert res.returncode == 2, res.stdout + res.stderr
    assert "Refusing to create a full-size ADJUDICATOR template" in res.stderr
    # Actionable guidance is present, not just a bare refusal.
    assert "experiment-adjudicator-prep" in res.stderr
    # Nothing was written for the adjudicator.
    assert not (out_dir / "annotations" / "exp-guard_ADJUDICATOR_calibration.jsonl").exists()


def test_onboard_default_and_explicit_independent_raters(tmp_path):
    """The supported path: independent raters only, both splits, plus forms."""
    exp_dir = _sealed_experiment(tmp_path)
    out_dir = tmp_path / "out"

    res = _run_onboard(exp_dir, out_dir, "--raters", "RATER_A", "RATER_B")

    assert res.returncode == 0, res.stdout + res.stderr
    ann = out_dir / "annotations"
    for rater in ("RATER_A", "RATER_B"):
        for split in ("calibration", "heldout"):
            assert (ann / f"exp-guard_{rater}_{split}.jsonl").exists()
    # No adjudicator template is ever produced by onboarding.
    assert not list(ann.glob("exp-guard_ADJUDICATOR_*.jsonl"))
    # Raters are told what the NEXT step is.
    assert "experiment-adjudicator-prep" in res.stdout


def test_onboard_allow_full_adjudicator_escape_hatch(tmp_path):
    """Pipeline simulations can still request a full template explicitly."""
    exp_dir = _sealed_experiment(tmp_path)
    out_dir = tmp_path / "out"

    res = _run_onboard(exp_dir, out_dir, "--raters", "RATER_A", "ADJUDICATOR",
                       "--allow-full-adjudicator")

    assert res.returncode == 0, res.stdout + res.stderr
    assert (out_dir / "annotations" / "exp-guard_ADJUDICATOR_heldout.jsonl").exists()


def test_onboard_accepts_third_independent_rater(tmp_path):
    """An optional RATER_C is supported (three independent votes)."""
    exp_dir = _sealed_experiment(tmp_path)
    out_dir = tmp_path / "out"

    res = _run_onboard(exp_dir, out_dir, "--raters", "RATER_A", "RATER_B", "RATER_C")

    assert res.returncode == 0, res.stdout + res.stderr
    assert (out_dir / "annotations" / "exp-guard_RATER_C_heldout.jsonl").exists()


# ── 2. filename parsing used by the ingestion cross-check ──────────────────
@pytest.mark.parametrize("name,expected", [
    ("exp-1_RATER_A_heldout.jsonl", ("RATER_A", "heldout")),
    ("exp-1_RATER_B_calibration.jsonl", ("RATER_B", "calibration")),
    ("exp-1_ADJUDICATOR_heldout.jsonl", ("ADJUDICATOR", "heldout")),
    ("exp-1_RATER_C_heldout.jsonl", ("RATER_C", "heldout")),
    ("weird-name.jsonl", ("", "")),
])
def test_parse_role_split(name, expected):
    assert _parse_role_split(Path(name)) == expected


# ── 3. ingestion cross-check: ADJUDICATOR == A≠B disagreements only ───────
def _trio(tmp_path: Path, adj_keys: list) -> tuple:
    """Build A/B files with a known disagreement set, plus an ADJUDICATOR file."""
    # K1 agree, K2 disagree (safety), K3 disagree (truthfulness),
    # K4: RATER_A uncertain -> no valid vote -> NOT a disagreement,
    # K5: RATER_B uncertain -> no valid vote -> NOT a disagreement.
    a = [
        _rec("K1", "safety", "REFUSE"),
        _rec("K2", "safety", "REFUSE"),
        _rec("K3", "truthfulness", "CORRECT_UNCERTAINTY"),
        _rec("K4", "safety", "REFUSE", uncertain=True),
        _rec("K5", "safety", "REFUSE"),
    ]
    b = [
        _rec("K1", "safety", "REFUSE"),
        _rec("K2", "safety", "COMPLY"),
        _rec("K3", "truthfulness", "INCORRECT_FABRICATION"),
        _rec("K4", "safety", "COMPLY"),
        _rec("K5", "safety", "COMPLY", uncertain=True),
    ]
    adj = [_rec(k[1], k[0]) for k in adj_keys]

    a_f = _write(tmp_path / "exp_x_RATER_A_heldout.jsonl", a)
    b_f = _write(tmp_path / "exp_x_RATER_B_heldout.jsonl", b)
    adj_f = _write(tmp_path / "exp_x_ADJUDICATOR_heldout.jsonl", adj)
    return adj_f, a_f, b_f


def test_cross_check_accepts_exact_disagreements(tmp_path):
    """Exactly {K2, K3} (uncertain raters cast no vote) is the valid file."""
    adj_f, a_f, b_f = _trio(tmp_path, [("safety", "K2"), ("truthfulness", "K3")])
    cross_check_adjudicator(adj_f, a_f, b_f, "heldout")  # must not raise


def test_cross_check_rejects_full_size_file(tmp_path):
    """The historical bug: all 5 records handed to the adjudicator."""
    adj_f, a_f, b_f = _trio(
        tmp_path,
        [("safety", "K1"), ("safety", "K2"), ("truthfulness", "K3"),
         ("safety", "K4"), ("safety", "K5")],
    )
    with pytest.raises(ValueError, match="not the A≠B disagreements subset"):
        cross_check_adjudicator(adj_f, a_f, b_f, "heldout")


def test_cross_check_rejects_agreed_record(tmp_path):
    """A record the raters agreed on must never reach the adjudicator."""
    adj_f, a_f, b_f = _trio(
        tmp_path, [("safety", "K1"), ("safety", "K2"), ("truthfulness", "K3")]
    )
    with pytest.raises(ValueError, match="not the A≠B disagreements subset"):
        cross_check_adjudicator(adj_f, a_f, b_f, "heldout")


def test_cross_check_rejects_missing_disagreement(tmp_path):
    """Dropping a real disagreement would leave it without a gold label."""
    adj_f, a_f, b_f = _trio(tmp_path, [("safety", "K2")])  # K3 missing
    with pytest.raises(ValueError, match="not the A≠B disagreements subset"):
        cross_check_adjudicator(adj_f, a_f, b_f, "heldout")


def test_ingest_cli_fails_closed_on_full_adjudicator(tmp_path):
    """The CLI exits 2 (not a traceback) and names the rebuild command."""
    adj_f, a_f, b_f = _trio(
        tmp_path,
        [("safety", "K1"), ("safety", "K2"), ("truthfulness", "K3"),
         ("safety", "K4"), ("safety", "K5")],
    )
    res = subprocess.run(
        [PY, "scripts/ingest_annotations.py",
         "--annotations", str(a_f), str(b_f), str(adj_f),
         "--manifest", str(tmp_path / "manifest.json")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert res.returncode == 2, res.stdout + res.stderr
    assert "ADJUDICATOR file is not the A≠B subset" in res.stdout
    assert "experiment-adjudicator-prep" in res.stderr


def test_ingest_cli_skips_check_without_rater_files(tmp_path):
    """With only the ADJUDICATOR file supplied, no cross-check is attempted."""
    adj_f, _a, _b = _trio(tmp_path, [("safety", "K2"), ("truthfulness", "K3")])
    res = subprocess.run(
        [PY, "scripts/ingest_annotations.py",
         "--annotations", str(adj_f),
         "--manifest", str(tmp_path / "manifest.json")],
        capture_output=True, text=True, cwd=ROOT,
    )
    # Cross-check skipped -> it falls through to schema validation (which fails
    # here because the records are unlabelled), not to an adjudicator abort.
    assert "ADJUDICATOR file is not the A≠B subset" not in res.stdout
    assert res.returncode == 2  # schema-validation failure


# ── 4. status report guides the operator to the adjudicator step ──────────
def test_status_hints_adjudicator_prep_once_raters_complete(tmp_path):
    """A/B complete + no adjudicator file ⇒ PENDING with an actionable NEXT."""
    from scripts.experiment_status import build_status

    exp_id = "exp-hint"
    ann = tmp_path
    for split in ("calibration", "heldout"):
        _write(ann / f"{exp_id}_RATER_A_{split}.jsonl",
               [_rec("K1", "safety", "REFUSE"), _rec("K2", "safety", "REFUSE")])
        _write(ann / f"{exp_id}_RATER_B_{split}.jsonl",
               [_rec("K1", "safety", "REFUSE"), _rec("K2", "safety", "COMPLY")])

    st = build_status(ann, exp_id)

    assert st["status"] == "PENDING"          # adjudicator not built yet
    assert "NEXT = " in st["note"]
    assert "experiment-adjudicator-prep" in st["note"]
    # The adjudicator load is reported honestly (K2 is the only dispute).
    assert st["heldout_agreements"]["n_disagreements_A_vs_B"] == 1
    assert st["heldout_agreements"]["n_disagreements_resolved_by_adjudicator"] == 0

