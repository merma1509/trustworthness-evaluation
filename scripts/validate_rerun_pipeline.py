#!/usr/bin/env python3
"""validate_rerun_pipeline.py — One-command clean-rerun validation gate.

Regenerates EVERY reported number of the human-validation experiment from the
**immutable** inputs (``results/raw_outputs`` + sealed templates + filled rater
annotations), then validates the headline invariants that make the rerun
scientifically trustworthy:

  * gold labels are re-derived ONLY from real annotation files
    (never fabricated from simulated raters);
  * every recorded annotation is accounted for (scored / unresolvable /
    explicitly excluded) — no silent drop;
  * agreement κ + CIs are within bounds and internally consistent;
  * template blindness holds on the sealed templates (no auto/model/prompt-id
    leakage) — via ``verify_seal_integrity``;
  * gold/auto join is complete (every gold key resolves to an auto label).

The script is intentionally THIN: it delegates the heavy lifting to the
existing single-responsibility drivers (``seal_experiment`` template blindness
is already enforced; ``resolve_disagreements`` re-checks coverage; ``compute_*``
regenerates scores/CIs/ranking; ``report_part1_agreement`` computes κ+CIs).
Here we only orchestrate and *fail closed* when any invariant breaks.

Usage:
    python3 scripts/validate_rerun_pipeline.py \
        --experiment-id trustworthiness-validation-2025-08-redo \
        [--annotations-dir experiment/annotations] \
        [--keep]   # do not delete the re-derived reports (default: restore)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action  # noqa: E402
from src.experiment import find_latest_experiment_id  # noqa: E402

# Headline invariants checked per dimension of the gold/agreement reports.
INTEGRITY_CHECKS = ("gold_vs_auto", "inter_rater", "ci", "coverage")


def _run(cmd: List[str], cwd: Path) -> int:
    """Run a subprocess, streaming output through, and return its exit code."""
    print(f"\n  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd))


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _fail(msg: str) -> None:
    print(f"\n  ✗ VALIDATION FAILED: {msg}")
    raise SystemExit(1)


def _check_coverage(resolve_report: Dict) -> None:
    """No-silent-drop invariant on the resolution report (fail closed)."""
    n_recorded = resolve_report.get("n_recorded")
    n_scored = resolve_report.get("n_scored")
    n_unres = resolve_report.get("n_unresolvable")
    n_excl = resolve_report.get("n_excluded_response_invalid")
    if n_recorded is None:
        _fail("resolution report missing 'n_recorded'")
    # An all-unresolvable report (0 scored) usually means blank templates or a
    # future refactor that stopped emitting labels — never bless it.
    if n_scored == 0 and n_recorded > 0:
        _fail(
            "resolution report scored 0 records (blank/unavailable annotations); "
            "refusing to validate a rerun with no gold labels"
        )
    if n_scored + n_unres + n_excl != n_recorded:
        _fail(
            f"coverage broken: scored({n_scored}) + unresolvable({n_unres}) + "
            f"excluded({n_excl}) != recorded({n_recorded})"
        )


def _check_agreement(agreement_path: Path) -> None:
    """Validate κ/agreement-rate/CI bounds from the held-out agreement report."""
    if not agreement_path.exists():
        _fail(f"agreement report not found: {agreement_path}")
    report = json.loads(agreement_path.read_text())

    gva = report.get("gold_vs_auto") or {}
    overall = gva.get("overall") or {}

    n = overall.get("n")
    kappa = overall.get("cohens_kappa")
    rate = overall.get("agreement_rate")
    if not isinstance(n, int) or n <= 0:
        _fail("gold_vs_auto/overall has no valid n")
    if kappa is not None and not (-1 <= kappa <= 1):
        _fail(f"overall κ out of bounds: {kappa}")
    if rate is not None and not (0 <= rate <= 1):
        _fail(f"overall agreement_rate out of bounds: {rate}")

    ci = overall.get("kappa_ci") or {}
    lo, up = ci.get("ci_lower"), ci.get("ci_upper")
    if lo is not None and up is not None and not (lo <= kappa <= up):
        _fail(f"overall κ ({kappa}) outside CI [{lo}, {up}]")

    # Inter-rater sections must individually be bounded too.
    for dim, stats in (report.get("inter_rater_A_vs_B") or {}).items():
        k = stats.get("cohens_kappa")
        if k is not None and not (-1 <= k <= 1):
            _fail(f"inter_rater/{dim} κ out of bounds: {k}")
        a = stats.get("agreement_rate")
        if a is not None and not (0 <= a <= 1):
            _fail(f"inter_rater/{dim} agreement_rate out of bounds: {a}")


def _check_gold_join(gold_path: Path) -> None:
    """Every gold record must carry an auto label (gold/auto join completeness)."""
    if not gold_path.exists():
        _fail(f"gold labels not found: {gold_path}")
    gold = _load_jsonl(gold_path)
    if not gold:
        _fail("gold_labels.jsonl is empty")
    missing_auto = [
        g.get("internal_key") for g in gold if not g.get("auto_label")
    ]
    if missing_auto:
        _fail(
            f"{len(missing_auto)} gold record(s) missing auto label "
            f"(join incomplete): {missing_auto[:5]}..."
        )


def validate_rerun(args) -> None:
    """Run the full regenerate + validate sequence; fail closed on any break."""
    root = Path.cwd()
    exp_id = args.experiment_id
    ann_dir = Path(args.annotations_dir)
    keep = args.keep

    # Guard: annotations must actually be present; otherwise we would be
    # "validating" the reproduce step that refuses to fabricate them.
    if not ann_dir.exists():
        _fail(f"annotations dir not found: {ann_dir}")
    rater_files = [
        ann_dir / f"{exp_id}_RATER_A_heldout.jsonl",
        ann_dir / f"{exp_id}_RATER_B_heldout.jsonl",
        ann_dir / f"{exp_id}_ADJUDICATOR_heldout.jsonl",
    ]
    missing = [str(p) for p in rater_files if not p.exists()]
    if missing:
        _fail("held-out annotation files missing: " + ", ".join(missing))
    for p in rater_files:
        if not _load_jsonl(p):
            _fail(f"held-out annotation file is empty: {p}")

    # Honesty guard: the annotation templates must actually be FILLED. A blank
    # template (all labels empty) yields n_scored=0 without any disagreement —
    # validating against it would silently bless an unfinished rerun.
    def _record_label(rec: dict) -> str:
        dim = (rec.get("prompt_record") or {}).get("dimension", "")
        ann = rec.get("annotations") or {}
        key = {"safety": "safety_label",
               "truthfulness": "truthfulness_label",
               "consistency": "consistency_label"}.get(dim, "")
        return (ann.get(key) or "").strip()

    all_labels = [_record_label(r) for f in rater_files for r in _load_jsonl(f)]
    filled = sum(1 for lb in all_labels if lb)
    if filled == 0:
        _fail(
            "annotation files contain NO filled labels (blank templates). "
            "Rerun experiment-annotate with the v3 rater instructions, have "
            "raters fill them, then re-ingest."
        )

    # ── 1. Verify the sealed experiment is intact & blinded ──────────────────
    print("\n=== 1/6 seal integrity & blindness ===")
    rc = _run(
        [
            sys.executable,
            "scripts/verify_seal_integrity.py",
            "--manifest", "experiment/sealed/manifests/sealing_manifest.json",
            "--sealed-dir", "experiment/sealed",
            "--sealed-labels",
            "experiment/sealed/labels/sealed_auto_labels.jsonl.enc",
        ],
        root,
    )
    if rc != 0:
        _fail("seal integrity verification failed")

    # ── 2. Resolve A/B disagreements with the ADJUDICATOR (held-out) ─────────
    print("\n=== 2/6 disagreement resolution (held-out) ===")
    resolve_out = Path(tempfile.mkdtemp(prefix="rerun_resolve_")) / "resolutions.json"
    rc = _run(
        [
            sys.executable,
            "scripts/resolve_disagreements.py",
            "--rater-a", str(rater_files[0]),
            "--rater-b", str(rater_files[1]),
            "--adjudicator", str(rater_files[2]),
            "--split", "heldout",
            "--experiment-id", exp_id,
            "--out", str(resolve_out),
        ],
        root,
    )
    if rc != 0:
        _fail("disagreement resolution failed")
    _check_coverage(json.loads(resolve_out.read_text()))

    # ── 3. Build gold labels + held-out agreement κ/CIs ──────────────────────
    print("\n=== 3/6 gold labels + agreement κ/CIs ===")
    gold_out = Path(tempfile.mkdtemp(prefix="rerun_gold_")) / "gold_labels.jsonl"
    rc = _run(
        [
            sys.executable,
            "scripts/generate_gold_labels.py",
            "--resolutions", str(resolve_out),
            "--sealed-labels",
            "experiment/sealed/labels/sealed_auto_labels.jsonl.enc",
            "--passphrase", args.passphrase or "",
            "--out", str(gold_out),
        ],
        root,
    )
    if rc != 0:
        _fail("gold label generation failed")
    _check_gold_join(gold_out)

    agreement_out = Path(tempfile.mkdtemp(prefix="rerun_agree_")) / "part1_agreement_report.json"
    rc = _run(
        [
            sys.executable,
            "scripts/report_part1_agreement.py",
            "--gold", str(gold_out),
            "--rater-a", str(rater_files[0]),
            "--rater-b", str(rater_files[1]),
            "--out", str(agreement_out),
            "--with-ci",
        ],
        root,
    )
    if rc != 0:
        _fail("agreement report generation failed")
    _check_agreement(agreement_out)

    # ── 4. Regenerate every reported NUMBER from immutable raw outputs ───────
    print("\n=== 4/6 reproduce dimension scores + TrustScore ===")
    rc = _run(
        [sys.executable, "scripts/compute_scores.py",
         "--raw", "results/raw_outputs", "--output",
         "results/scores_report.json"],
        root,
    )
    rc |= _run(
        [sys.executable, "scripts/generate_results_json.py",
         "--scores", "results/scores_report.json",
         "--output-results", "results"],
        root,
    )
    if rc != 0:
        _fail("scores/trustscore regeneration failed")

    print("\n=== 5/6 reproduce CIs + ranking stability ===")
    rc = _run(
        [sys.executable, "scripts/compute_ci.py",
         "--raw", "results/raw_outputs", "--n-bootstrap", "10000",
         "--output", "results/ci_report.json"],
        root,
    )
    rc |= _run(
        [sys.executable, "scripts/compute_ranking_stability.py",
         "--scores", "results/scores_report.json", "--n-bootstrap", "10000",
         "--output", "results/ranking_stability.json"],
        root,
    )
    if rc != 0:
        _fail("CI/ranking regeneration failed")

    # ── 6. Verify internal consistency of the regenerated reports ───────────
    print("\n=== 6/6 internal consistency verification ===")
    rc = _run(
        [sys.executable, "scripts/verify_results.py",
         "--results", "results",
         "--expected", "results/expected_results.json"],
        root,
    )
    if rc != 0:
        _fail("regenerated reports are internally inconsistent")

    if keep:
        # Persist the freshly derived agreement artifacts so a later
        # `experiment-agreement` / dashboard read them.
        out_agreements = Path("experiment/agreements")
        out_gold = Path("experiment/gold")
        out_reports = Path("experiment/reports")
        out_agreements.mkdir(parents=True, exist_ok=True)
        out_gold.mkdir(parents=True, exist_ok=True)
        out_reports.mkdir(parents=True, exist_ok=True)
        resolve_dest = out_agreements / f"{exp_id}_heldout_disagreements.json"
        gold_dest = out_gold / "gold_labels.jsonl"
        agree_dest = out_reports / "part1_agreement_report.json"
        resolve_dest.write_bytes(resolve_out.read_bytes())
        gold_dest.write_bytes(gold_out.read_bytes())
        agree_dest.write_bytes(agreement_out.read_bytes())

    print("\n  ✓ VALIDATION PASSED — every reported number regenerated "
          "from immutable inputs and internally consistent.")

    log_action(
        log_path=Path("experiment/logs/processing_log.jsonl"),
        action="validate_rerun_pipeline",
        script="scripts/validate_rerun_pipeline.py",
        args={"--experiment-id": exp_id, "--annotations-dir": str(ann_dir)},
        input_paths={"annotations_dir": ann_dir,
                     "sealed_dir": Path("experiment/sealed"),
                     "raw": Path("results/raw_outputs")},
        output_paths={"scores": Path("results/scores_report.json"),
                      "ci": Path("results/ci_report.json"),
                      "ranking": Path("results/ranking_stability.json")},
        outcome="ok",
        status=0,
        extra={"checks": list(INTEGRITY_CHECKS)},
    )


def main() -> int:
    default_exp_id = (
        find_latest_experiment_id()
        or "trustworthiness-validation-0000-00-00-000000"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=default_exp_id)
    parser.add_argument("--annotations-dir", default="experiment/annotations")
    parser.add_argument("--passphrase", default=None,
                        help="Passphrase to open sealed auto labels for the gold join.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep derived agreement/gold reports under experiment/.")
    args = parser.parse_args()

    if not args.passphrase:
        # The reproduce pipeline is designed to FAIL CLOSED rather than
        # fabricate these numbers when the passphrase is unavailable.
        print("  Provide --passphrase to unseal the auto labels for the gold join.")
        return 2

    try:
        validate_rerun(args)
    except SystemExit as e:
        log_action(
            log_path=Path("experiment/logs/processing_log.jsonl"),
            action="validate_rerun_pipeline",
            script="scripts/validate_rerun_pipeline.py",
            args={"--experiment-id": args.experiment_id},
            outcome="failed",
            status=int(e.code or 1),
            extra={"error": str(e)},
        )
        return int(e.code or 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
