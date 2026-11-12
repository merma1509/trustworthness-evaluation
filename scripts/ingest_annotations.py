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

from src.audit_log import log_action
from src.labels import (
    is_valid_label,
)
from src.sealing import sha256_file

# Which ``annotations.*`` field carries the label for each dimension.
_DIM_LABEL_KEY = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}


# Validation
def _dimension_of(rec: dict) -> str:
    """Dimension lives in ``prompt_record.dimension`` for sealed templates"""
    pr = rec.get("prompt_record") or {}
    return (pr.get("dimension") or rec.get("dimension") or "").strip()


def validate_annotation_record(rec: dict) -> Dict:
    """Validate one annotation record against the constrained schema.

    Returns a dict with ``valid`` plus detailed status. ``notes`` (free text) is
    optional but the label MUST be a finite-set value.

    Abstention is a legitimate outcome (rubric §3 "no skip / no exclusion"): a
    rater who cannot decide sets ``rater_uncertainty = "uncertain"`` and leaves
    the label blank. That record is VALID and counts as uncertainty data — it is
    never forced into an arbitrary label and never dropped from the dataset.
    A blank label WITHOUT the uncertain flag is a genuine omission and is invalid.
    """
    dim = _dimension_of(rec)
    ann = rec.get("annotations") or {}
    flags = rec.get("flags") or {}

    # Determine the label actually used for this dimension
    label = ann.get(_DIM_LABEL_KEY.get(dim, ""), "")

    problems: List[str] = []

    if not dim:
        problems.append("missing dimension")

    # rater_uncertainty lives under ``flags``
    uncertainty = flags.get("rater_uncertainty", "")
    if uncertainty and uncertainty not in ("confident", "uncertain", "guess"):
        problems.append(f"invalid rater_uncertainty '{uncertainty}'")

    is_abstention = uncertainty == "uncertain" and not label

    # Label-constrained check: a declared abstention needs no label; anything
    # else must carry exactly one label from the finite set for its dimension.
    if not label and not is_abstention:
        problems.append("no label assigned (must pick exactly one)")
    elif label and dim and not is_valid_label(dim, label):
        problems.append(
            f"invalid label '{label}' for dimension '{dim}' "
            f"(not in constrained set)"
        )

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
        "abstained": is_abstention,
        "response_invalid": bool(flags.get("response_invalid")),
    }


def _record_key(rec: dict) -> tuple:
    """Join key shared with resolve_disagreements/prepare_adjudicator."""
    dim = _dimension_of(rec)
    ik = rec.get("internal_key", "")
    if ik:
        return (dim, ik)
    pr = rec.get("prompt_record") or {}
    return (dim, f"pid::{pr.get('prompt_id', '')}")


def _effective_label(rec: dict):
    """Rater's vote for this record, or None when absent/uncertain.

    Mirrors the "uncertain raters cast no valid vote" rule in
    ``resolve_disagreements.py`` so the disagreement set matches downstream.
    """
    dim = _dimension_of(rec)
    label = (rec.get("annotations") or {}).get(_DIM_LABEL_KEY.get(dim, ""), "")
    if not label:
        return None
    if (rec.get("flags") or {}).get("rater_uncertainty") == "uncertain":
        return None
    return label


def _load_by_key(path: Path) -> Dict[tuple, dict]:
    return {
        _record_key(r): r
        for r in (json.loads(line) for line in path.open() if line.strip())
    }


def cross_check_adjudicator(adj_path: Path, a_path: Path, b_path: Path,
                            split: str) -> None:
    """Fail closed unless the ADJUDICATOR file holds exactly the A≠B records.

    This is the precise guard for the 147/63-record bug: an adjudicator file
    that still contains the full sealed set (or any extra/missing record) is
    rejected with an actionable message.
    """
    a = _load_by_key(a_path)
    b = _load_by_key(b_path)
    adj = _load_by_key(adj_path)

    # A≠B = both cast a valid (non-uncertain) vote that differs.
    expected = {
        key for key in set(a) & set(b)
        if _effective_label(a[key]) and _effective_label(b[key])
        and _effective_label(a[key]) != _effective_label(b[key])
    }
    got = set(adj)

    extra = got - expected      # adjudicator was shown records with no dispute
    missing = expected - got    # real disputes never reached the adjudicator

    if extra or missing:
        msg = [
            f"  ✗ {adj_path.name}: ADJUDICATOR file does not match the A≠B "
            f"disagreements ({split}).",
            f"    expected {len(expected)} disagreement record(s); file has "
            f"{len(got)}.",
        ]
        if extra:
            sample = ", ".join(f"{d}:{k}" for d, k in sorted(extra)[:5])
            msg.append(f"    {len(extra)} record(s) the raters AGREED on "
                       f"(must not be adjudicated): {sample}")
        if missing:
            sample = ", ".join(f"{d}:{k}" for d, k in sorted(missing)[:5])
            msg.append(f"    {len(missing)} real disagreement(s) MISSING "
                       f"(no gold label possible): {sample}")
        msg.append("    Rebuild it with:  make experiment-adjudicator-prep "
                   f"SPLIT={split}")
        print("\n".join(msg), file=sys.stderr)
        raise ValueError(
            f"{adj_path.name}: adjudicator file is not the A≠B disagreements "
            f"subset (expected {len(expected)}, got {len(got)})"
        )


def _parse_role_split(path: Path) -> tuple:
    """Extract ``(role, split)`` from ``{exp}_{ROLE}_{split}.jsonl``.

    Returns ``("", "")`` when the name does not follow the convention (the
    cross-check is then skipped for that file rather than guessed).
    """
    stem = path.stem  # e.g. exp-1_RATER_A_heldout
    for role in ("ADJUDICATOR", "RATER_A", "RATER_B", "RATER_C"):
        marker = f"_{role}_"
        if marker in stem:
            return role, stem.split(marker, 1)[1]
    return "", ""


def _enforce_adjudicator_disagreements_only(files: List[Path]) -> None:
    """Reject ADJUDICATOR files that are not exactly the A≠B disagreements.

    Runs only when the matching RATER_A and RATER_B files are supplied in the
    same invocation; otherwise it is silent (there is nothing to compare to).
    Raises ``ValueError`` on mismatch — ingestion then fails closed.
    """
    by_key = {}
    for p in files:
        role, split = _parse_role_split(p)
        if role and split:
            by_key[(split, role)] = p

    for (split, role), adj_path in sorted(by_key.items()):
        if role != "ADJUDICATOR":
            continue
        a_path = by_key.get((split, "RATER_A"))
        b_path = by_key.get((split, "RATER_B"))
        if not (a_path and b_path):
            print(f"  (note) {adj_path.name}: skipped A≠B cross-check "
                  f"(RATER_A/RATER_B for '{split}' not supplied)")
            continue
        cross_check_adjudicator(adj_path, a_path, b_path, split)


def ingest_file(path: Path) -> Dict:
    """Validate every record in one annotation file; return a stats summary."""
    records = [json.loads(line) for line in path.open() if line.strip()]
    results = [validate_annotation_record(r) for r in records]

    n_valid = sum(1 for r in results if r["valid"])
    n_uncertain = sum(1 for r in results if r["uncertain"])
    n_abstained = sum(1 for r in results if r["abstained"])
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
        "n_abstained": n_abstained,
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

    # ── ADJUDICATOR cross-check (prevents the full-size-template bug) ────────
    # Group the supplied files by (split, role) parsed from the filename pattern
    # {experiment}_{ROLE}_{split}.jsonl, then require every ADJUDICATOR file to
    # contain exactly the A≠B disagreements for its split. Fails closed.
    try:
        _enforce_adjudicator_disagreements_only(files)
    except ValueError:
        log_action(
            log_path=Path("experiment/logs/annotation_log.jsonl"),
            action="ingest_annotations",
            script="scripts/ingest_annotations.py",
            args={"--annotations": args.annotations, "--manifest": args.manifest},
            input_paths={"annotations": files},
            outcome="failed",
            status=2,
            extra={"reason": "adjudicator_not_disagreements_only"},
        )
        print("\n  Ingestion aborted: ADJUDICATOR file is not the A≠B subset.")
        return 2

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
              f"abstained={rep['n_abstained']} "
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
        log_action(
            log_path=Path("experiment/logs/annotation_log.jsonl"),
            action="ingest_annotations",
            script="scripts/ingest_annotations.py",
            args={"--annotations": args.annotations, "--manifest": args.manifest},
            input_paths={"annotations": files},
            outcome="failed",
            status=2,
            extra={"all_valid": False},
        )
        return 2
    print("  All annotations are schema-valid.")

    # Audit trail: filing annotations is immutable and
    # timestamped; log the accepted files and their per-file validity counts.
    log_action(
        log_path=Path("experiment/logs/annotation_log.jsonl"),
        action="ingest_annotations",
        script="scripts/ingest_annotations.py",
        args={"--annotations": args.annotations, "--manifest": args.manifest},
        input_paths={"annotations": files},
        output_paths={"annotation_manifest": out_path},
        extra={"all_valid": True,
               "file_summary": [
                   {"file": r.get("file"),
                    "n_records": r.get("n_records"), "n_valid": r.get("n_valid"),
                    "n_invalid": r.get("n_invalid")}
                   for r in reports
               ]},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


