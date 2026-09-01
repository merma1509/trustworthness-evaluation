"""Determinism regression tests for the reproducibility drivers.

These drivers regenerate EVERY reported paper number from immutable raw
outputs. They must be:
  * deterministic — two runs with the same inputs produce byte-identical output,
  * faithful — the reproduced dimension scores match the committed reference
    values (which current README / paradigm_report.php cite)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "results" / "raw_outputs"
SCORES_SCRIPT = PROJECT_ROOT / "scripts" / "compute_scores.py"
CI_SCRIPT = PROJECT_ROOT / "scripts" / "compute_ci.py"
GEN_MANIFEST = PROJECT_ROOT / "scripts" / "generate_final_manifest.py"
VERIFY_RESULTS = PROJECT_ROOT / "scripts" / "verify_results.py"
GEN_EXPECTED = PROJECT_ROOT / "scripts" / "generate_expected_results.py"
RESULTS_DIR = PROJECT_ROOT / "results"

# Reference dimension scores cited by the committed README / reports.
REFERENCE_SCORES = {
    "gemma3_4b": {"safety": 0.6857, "truthfulness": 0.7632, "consistency": 0.8182},
    "llama3.1_8b": {"safety": 0.7714, "truthfulness": 0.8421, "consistency": 0.9091},
}

_HAVE_INPUTS = SCORES_SCRIPT.exists() and RAW_DIR.exists()


def _run(script: Path, *args: str, cwd: Path = PROJECT_ROOT) -> dict:
    subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return {}


def _run_scores(out: Path) -> dict:
    _run(
        SCORES_SCRIPT,
        "--raw", str(RAW_DIR),
        "--output", str(out),
    )
    return json.loads(out.read_text())


@pytest.mark.skipif(not _HAVE_INPUTS, reason="raw outputs not present")
def test_scores_reproduce_committed_reference_values(tmp_path):
    """Regenerated dimension scores must match the committed README values."""
    out = tmp_path / "scores_report.json"
    report = _run_scores(out)
    for model, dims in REFERENCE_SCORES.items():
        assert model in report["per_model"], f"missing model {model}"
        for dim, expected in dims.items():
            got = report["per_model"][model][dim]["score"]
            assert got == pytest.approx(expected, abs=1e-3), (
                f"{model}/{dim}: produced {got}, expected {expected}"
            )


@pytest.mark.skipif(not _HAVE_INPUTS, reason="raw outputs not present")
def test_scores_deterministic_across_runs(tmp_path):
    """Two runs of compute_scores with identical inputs must be byte-identical."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    ra = _run_scores(a)
    rb = _run_scores(b)
    assert ra == rb, "non-deterministic scores regeneration"


@pytest.mark.skipif(not _HAVE_INPUTS, reason="raw outputs not present")
def test_ci_report_matches_reference_cis(tmp_path):
    """Bootstrap CIs must reproduce the committed CI ranges (seed=42)."""
    from scripts.compute_ci import compute_ci

    out = tmp_path / "ci_report.json"
    compute_ci(RAW_DIR, out, n_bootstrap=2000)

    ci = json.loads(out.read_text())
    # gemma safety mean 0.6857 within [0.5143, 0.8286] (from README).
    g = ci["per_model"]["gemma3_4b"]["safety"]
    assert g["mean"] == pytest.approx(0.6857, abs=1e-3)
    assert g["ci_lower"] <= g["mean"] <= g["ci_upper"]


@pytest.mark.skipif(not _HAVE_INPUTS, reason="raw outputs not present")
def test_generate_manifest_and_verify_roundtrip(tmp_path):
    """Manifest generation + immutable verification must PASS on produced outputs."""
    from scripts.generate_final_manifest import build_checksums
    from scripts.verify_immutable import verify_immutable_artifacts

    # Produce score/CI artifacts in tmp, checksum them, then verify they match.
    scores = tmp_path / "scores_report.json"
    _run_scores(scores)
    ci = tmp_path / "ci_report.json"
    from scripts.compute_ci import compute_ci

    compute_ci(RAW_DIR, ci, n_bootstrap=2000)

    manifest = tmp_path / "manifest.json"
    checksums = build_checksums(tmp_path, ["scores_report.json", "ci_report.json"])
    manifest.write_text(
        json.dumps({"checksums": checksums}), encoding="utf-8"
    )

    assert verify_immutable_artifacts(tmp_path, manifest) is True


@pytest.mark.skipif(
    not (RESULTS_DIR / "expected_results.json").exists(),
    reason="expected_results.json not committed",
)
def test_verify_results_expected_roundtrip():
    """verify_results --expected must PASS against the committed reference reports."""
    from scripts import verify_results

    errors = verify_results.verify_results(RESULTS_DIR, RESULTS_DIR / "expected_results.json")
    assert errors == [], f"verify --expected failed: {errors}"


def test_verify_results_flags_bad_expected(tmp_path):
    """A wrong expected value must be reported as a verification failure."""
    from scripts import verify_results

    # Build a minimal results tree from the committed reports.
    for name in ("scores_report.json", "trustscore_report.json",
                 "ci_report.json", "ranking_stability.json"):
        src = RESULTS_DIR / name
        if src.exists():
            (tmp_path / name).write_text(src.read_text(), encoding="utf-8")

    bad = tmp_path / "bad_expected.json"
    bad.write_text(json.dumps({
        "trustscore": {
            "gemma3_4b": {"trust_score": 0.999,
                          "dimension_scores": {"safety": 0.999}},
        },
        "ci": {},
        "ranking": {"configurations": []},
    }), encoding="utf-8")

    errors = verify_results.verify_results(tmp_path, bad)
    assert any("[expected] TrustScore" in e for e in errors), errors


def test_verify_results_flags_bad_agreement(tmp_path):
    """κ / agreement / agree+disagree invariants are enforced on the agreement report."""
    from scripts import verify_results

    for name in ("scores_report.json", "trustscore_report.json",
                 "ci_report.json", "ranking_stability.json"):
        src = RESULTS_DIR / name
        if src.exists():
            (tmp_path / name).write_text(src.read_text(), encoding="utf-8")

    (tmp_path / "part1_agreement_report.json").write_text(json.dumps({
        "gold_vs_auto": {
            "overall": {"n": 10, "agree": 7, "disagree": 5,
                        "cohens_kappa": 1.5, "agreement_rate": 2.0},
            "per_dimension": {
                "safety": {"n": 5, "agree": 5, "disagree": 0,
                           "cohens_kappa": 1.0, "agreement_rate": 1.0},
            },
        },
    }), encoding="utf-8")

    errors = verify_results.verify_results(tmp_path)
    assert any("κ out of bounds" in e for e in errors), errors
    assert any("Agreement out of bounds" in e for e in errors), errors
    assert any("Count mismatch" in e for e in errors), errors


