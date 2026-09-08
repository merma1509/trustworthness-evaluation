#!/usr/bin/env python3
"""experiment_status.py — Report whether the sealed human annotations are ready.

This is the honesty gate for ``make experiment-reproduce``: it inspects the
per-rater sealed annotation files (RATER_A / RATER_B / ADJUDICATOR, calibration +
heldout) and reports, per file:

    * how many records carry a (label-constrained) annotation,
    * whether every label is valid for its dimension,
    * how many A-vs-B disagreements exist (those the ADJUDICATOR must resolve),

and finally a single STATUS:

    * ``READY``   — every sealed annotation file is fully filled and schema-valid
                    (adjudicator included), so the full pipeline may run.
    * ``PENDING`` — some annotations are missing/invalid; the pipeline must NOT
                    emit a fabricated result until they are provided by real
                    independent raters.

It NEVER edits any annotation file; it only reads and reports.

Usage:
    python3 scripts/experiment_status.py \
        --annotations-dir experiment/annotations \
        --experiment-id trustworthiness-validation-2025-08-redo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Same constrained-vocabulary source of truth as the rest of the pipeline.
try:
    from src.labels import is_valid_label
except Exception:  # pragma: no cover - fallback for standalone invocation
    is_valid_label = None


DIMENSION_LABEL_KEY = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}


def _dimension_of(rec: dict) -> str:
    pr = rec.get("prompt_record") or {}
    return (pr.get("dimension") or "").strip()


def _label_of(rec: dict) -> str:
    dim = _dimension_of(rec)
    key = DIMENSION_LABEL_KEY.get(dim)
    if not key:
        return ""
    return (rec.get("annotations") or {}).get(key, "") or ""


def inspect_file(path: Path) -> Dict:
    """Return fill/invalid counts and per-record validity for one file."""
    records = [json.loads(line) for line in path.open() if line.strip()]
    filled = 0
    invalid = 0
    missing = 0
    missing_records = []
    for i, rec in enumerate(records):
        label = _label_of(rec)
        dim = _dimension_of(rec)
        if not label:
            missing += 1
            missing_records.append({"index": i, "dimension": dim, "internal_key": rec.get("internal_key")})
            continue
        filled += 1
        if dim and is_valid_label is not None and not is_valid_label(dim, label):
            invalid += 1
    return {
        "file": str(path),
        "n_records": len(records),
        "n_filled": filled,
        "n_missing": missing,
        "n_invalid": invalid,
        "missing_records": missing_records[:10],
    }


def _agreements_across(files) -> Dict:
    """Count A-vs-B disagreements across the heldout files (adjudicator load).

    Bodies are keyed by internal_key -> (dimension, label).
    """
    def _load(path) -> Dict[str, tuple]:
        out = {}
        if not path.exists():
            return out
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec.get("internal_key", "")] = (_dimension_of(rec), _label_of(rec))
        return out

    specs = [files["RATER_A"], files["RATER_B"], files["ADJUDICATOR"]]
    tables = [_load(p) for p in specs]
    common = set(tables[0]) & set(tables[1])
    disagreement_keys = [
        k for k in common
        if tables[0][k][1] and tables[1][k][1] and tables[0][k][1] != tables[1][k][1]
    ]
    adjudicator_resolved = sum(
        1 for k in disagreement_keys
        if tables[2].get(k, ("", ""))[1]
    )
    return {
        "n_common": len(common),
        "n_disagreements_A_vs_B": len(disagreement_keys),
        "n_disagreements_resolved_by_adjudicator": adjudicator_resolved,
    }


def build_status(annotations_dir: Path, experiment_id: str) -> Dict:
    per_file: List[Dict] = []
    any_invalid = False
    any_missing = False
    status = "READY"

    by_split_rater = {}
    for split in ("calibration", "heldout"):
        by_split_rater.setdefault(split, {})
        for rater in ("RATER_A", "RATER_B", "ADJUDICATOR"):
            p = annotations_dir / f"{experiment_id}_{rater}_{split}.jsonl"
            rep = None if not p.exists() else inspect_file(p)
            if rep is None:
                any_missing = True
                per_file.append({"file": str(p), "n_records": 0, "n_filled": 0,
                                 "n_missing": 0, "n_invalid": 0, "note": "file not found"})
                by_split_rater[split][rater] = p
                continue
            per_file.append(rep)
            any_invalid = any_invalid or rep["n_invalid"] > 0
            any_missing = any_missing or rep["n_missing"] > 0
            by_split_rater[split][rater] = p

    agreements = _agreements_across(by_split_rater.get("heldout", {}))

    if any_invalid or any_missing:
        status = "PENDING"

    return {
        "experiment_id": experiment_id,
        "status": status,
        "per_file": per_file,
        "heldout_agreements": agreements,
        "note": (
            "READY  = every rater file fully filled & schema-valid (adjudicator "
            "included), so the full pipeline may run.\n"
            "PENDING = some annotations missing/invalid — emit NO fabricated "
            "result until real independent raters provide them."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", default="experiment/annotations")
    parser.add_argument("--experiment-id", default="trustworthiness-validation-2025-08-redo")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print only the STATUS token (READY|PENDING) on stdout — for shell gating.",
    )
    parser.add_argument(
        "--fail-pending",
        action="store_true",
        help="Exit non-zero when STATUS != READY. Lets a Makefile gate 'fail closed' "
             "instead of fabricating a result from simulated/short annotations.",
    )
    args = parser.parse_args()

    status = build_status(Path(args.annotations_dir), args.experiment_id)

    if args.status_only:
        # Machine-readable: emit only the token so `make` can branch on it.
        print(status["status"])
        return 0 if status["status"] == "READY" else (1 if args.fail_pending else 0)

    print(f"\n  EXPERIMENT ANNOTATION STATUS: {status['status']}\n")
    for f in status["per_file"]:
        note = f.get("note", "")
        if note:
            print(f"    {f['file'].split('/')[-1]:<58} MISSING ({note})")
            continue
        problems = []
        if f["n_missing"]:
            problems.append(f"{f['n_missing']} unfilled")
        if f["n_invalid"]:
            problems.append(f"{f['n_invalid']} invalid labels")
        extra = ("  [" + ", ".join(problems) + "]") if problems else ""
        print(
            f"    {f['file'].split('/')[-1]:<58} {f['n_filled']}/{f['n_records']} filled{extra}"
        )

    ag = status["heldout_agreements"]
    print(f"\n  Held-out A-vs-B: {ag['n_disagreements_A_vs_B']} disagreements "
          f"({ag['n_disagreements_resolved_by_adjudicator']} resolved by adjudicator)")

    if status["status"] == "READY":
        print("\n  → Ready to run ingest → resolve → gold → agreement.\n")
    else:
        print("\n  → PENDING: real raters must fill the sealed templates first.\n")

    # In the default (report) mode we exit 0 even when PENDING, because this is
    # a status report. The `--fail-pending` flag is what lets a Makefile step
    # fail closed on the annotation-derived figure.
    if args.fail_pending and status["status"] != "READY":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

