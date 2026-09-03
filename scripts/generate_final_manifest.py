#!/usr/bin/env python3
"""generate_final_manifest.py — write results/manifest.json with checksums.

Implements the manifest-generation. Records a
SHA-256 prefix for every reproducibility artifact — immutable inputs (sealed
templates, annotations, gold, sealed labels) and generated outputs (scores,
CIs, ranking stability, cost) — so that ``verify_immutable.py`` can later
confirm nothing drifted.

Usage:
    python3 scripts/generate_final_manifest.py \
        --output results/manifest.json \
        --base-dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action  # noqa: E402
from src.sealing import sha256_file  # noqa: E402


def _existing(path: Path) -> bool:
    """True if the path exists; print a warning if it was requested but absent."""
    if path.exists():
        return True
    return False


def build_checksums(base_dir: Path, artifacts: List[str]) -> Dict[str, str]:
    """Compute a sha256:<prefix> for each artifact path (absolute-agnostic)."""
    checksums: Dict[str, str] = {}
    for rel in artifacts:
        p = base_dir / rel
        if not p.exists():
            print(f"  (note) artifact absent, skipped from manifest: {rel}")
            continue
        digest = sha256_file(p)
        checksums[rel] = f"sha256:{digest[:16]}"
    return checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="Where to write results/manifest.json.")
    parser.add_argument("--base-dir", default=".",
                        help="Root dir anchoring manifest-relative paths (default: .).")
    # Repeatable --artifact rel/path.json entries. If none are given, a sensible
    # default set covering the clean-redo + reproducibility pipeline is used.
    parser.add_argument("--artifact", action="append", default=[],
                        help="Relative artifact path to checksum (repeatable).")
    args = parser.parse_args()

    base = Path(args.base_dir)

    DEFAULT_ARTIFACTS = [
        # Immutable sealed inputs
        "experiment/sealed/manifests/sealing_manifest.json",
        "experiment/sealed/labels/sealed_auto_labels.jsonl.enc",
        # Gold + agreement (reproducibility checkpoints)
        "experiment/gold/gold_labels.jsonl",
        "experiment/reports/part1_agreement_report.json",
        # Generated score outputs
        "results/scores_report.json",
        "results/ci_report.json",
        "results/trustscore_report.json",
        "results/ranking_stability.json",
        "results/cost_tracker.json",
    ]

    artifacts = args.artifact or DEFAULT_ARTIFACTS

    checksums = build_checksums(base, artifacts)
    manifest = {
        "pipeline": "clean-redo reproducibility",
        "generator": "scripts/generate_final_manifest.py",
        "n_artifacts": len(checksums),
        "checksums": checksums,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  Manifest written: {out_path}")
    print(f"  {len(checksums)} artifact checksums recorded:")
    for rel in sorted(checksums):
        print(f"    {checksums[rel]}  {rel}")

    # Audit trail of the manifest generation process: log the write so manifest provenance
    # is itself on the audit trail.
    log_action(
        log_path=Path("experiment/logs/processing_log.jsonl"),
        action="generate_final_manifest",
        script="scripts/generate_final_manifest.py",
        args={"--output": args.output, "--base-dir": args.base_dir,
              "--artifact": args.artifact or DEFAULT_ARTIFACTS},
        output_paths={"manifest": out_path},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
