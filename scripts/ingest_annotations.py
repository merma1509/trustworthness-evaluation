#!/usr/bin/env python3
"""ingest_annotations.py — Validate + register rater annotations

This script is the label-constrained gate:
    * every annotation record MUST carry a label that is exactly one of the
      finite set for its dimension (see ``src/labels.py``),
    * ``rater_uncertainty`` must be one of ``confident|uncertain|guess``,
    * ``ai_assistance_used`` must be ``false`` for the annotation to be VALID,
    * if a rater could not decide, ``rater_uncertainty=uncertain`` is accepted
      (and counted as uncertainty data — it is NOT dropped)

It writes a per-rater validation report and updates the annotation manifest

Usage:
    python3 scripts/ingest_annotations.py \
        --annotations experiment/annotations/{exp}_{RATER}_{split}.jsonl ...
        --manifest experiment/manifests/annotation_manifest.json

Output:
    *_validation.json report per annotation file,
    updated annotation_manifest.json with counts + integrity checksums
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labels import (
    is_valid_label,
)
from src.sealing import sha256_file


# Validation
def _dimension_of(rec: dict) -> str:
    """Dimension lives in ``prompt_record.dimension`` for sealed templates"""
    pr = rec.get("prompt_record") or {}
    return (pr.get("dimension") or rec.get("dimension") or "").strip()


def validate_annotation_record(rec: dict) -> Dict:
    """Validate one annotation record against the constrained schema.

    Returns a dict with ``valid`` plus detailed status. ``notes`` (free text) is
    optional but the label MUST be a finite-set value.
    """
    dim = _dimension_of(rec)
    ann = rec.get("annotations") or {}

    # Determine the label actually used for this dimension
    label_key = {
        "safety": "safety_label",
        "truthfulness": "truthfulness_label",
        "consistency": "consistency_label",
    }.get(dim)
    label = ann.get(label_key, "") if label_key else ""

    problems: List[str] = []

    if not dim:
        problems.append("missing dimension")

    dim = _dimension_of(rec)
    ann = rec.get("annotations") or {}
    flags = rec.get("flags") or {}

    # Determine the label actually used for this dimension
    label_key = {
        "safety": "safety_label",
        "truthfulness": "truthfulness_label",
        "consistency": "consistency_label",
    }.get(dim)
    label = ann.get(label_key, "") if label_key else ""

    problems: List[str] = []

    if not dim:
        problems.append("missing dimension")

    # Label-constrained check
    if not label:
        problems.append("no label assigned (must pick exactly one)")
    elif not is_valid_label(dim, label):
        problems.append(
            f"invalid label '{label}' for dimension '{dim}' "
            f"(not in constrained set)"
        )

    # rater_uncertainty lives under ``flags``
    uncertainty = flags.get("rater_uncertainty", "")
    if uncertainty and uncertainty not in ("confident", "uncertain", "guess"):
        problems.append(f"invalid rater_uncertainty '{uncertainty}'")

    ai_used = flags.get("ai_assistance_used")
    if ai_used not in (False, None, ""):
        problems.append(
            "annotation invalid: AI assistance used (must be declared & re-done)"
        )

    return {
        "dimension": dim,
        "label": label,
        "valid": not problems,
        "problems": problems,
        "uncertain": uncertainty == "uncertain",
        "response_invalid": bool(flags.get("response_invalid")),
    }


def ingest_file(path: Path) -> Dict:
    """Validate every record in one annotation file; return a stats summary."""
    records = [json.loads(l) for l in path.open() if l.strip()]
    results = [validate_annotation_record(r) for r in records]

    n_valid = sum(1 for r in results if r["valid"])
    n_uncertain = sum(1 for r in results if r["uncertain"])
    n_response_invalid = sum(1 for r in results if r["response_invalid"])
    n_invalid = len(records) - n_valid
    invalid_examples = [
        {"index": i, **r} for i, r in enumerate(results) if not r["valid"]
    ][:10]

    return {
        "file": str(path),
        "n_records": len(records),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_uncertain": n_uncertain,
        "n_response_invalid": n_response_invalid,
        "invalid_examples": invalid_examples,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", nargs="+", required=True,
        help="One filled annotation JSONL per rater/split",
    )
    parser.add_argument("--manifest", default="experiment/manifests/annotation_manifest.json")
    parser.add_argument("--declarations-dir", default="experiment/rater_declarations")
    args = parser.parse_args()

    files = [Path(p) for p in args.annotations]
    if not files:
        print("  No annotation files given.")
        return 1

    reports = []
    all_valid = True
    for p in files:
        if not p.exists():
            print(f"  Annotation file not found: {p}")
            all_valid = False
            continue
        rep = ingest_file(p)
        reports.append(rep)
        status = "OK" if rep["n_invalid"] == 0 else f"FAIL ({rep['n_invalid']} invalid)"
        print(f"  {p.name:<60} {status}  n={rep['n_records']} "
              f"valid={rep['n_valid']} uncertain={rep['n_uncertain']} "
              f"resp_invalid={rep['n_response_invalid']}")
        if rep["invalid_examples"]:
            for ex in rep["invalid_examples"]:
                print(f"      - idx {ex['index']}: {ex['problems']}")
        if rep["n_invalid"]:
            all_valid = False

    # Write a combined validation report alongside the manifest
    out_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "all_valid": all_valid,
        "files": reports,
    }
    with out_path.open("w") as f:
        json.dump(doc, f, indent=2)

    print(f"\n  Combined validation -> {out_path}")
    if not all_valid:
        print("  One or more annotation files have schema violations.")
        print("    Fix them (labels MUST be from the constrained set) before proceeding.")
        return 2
    print("  All annotations are schema-valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
