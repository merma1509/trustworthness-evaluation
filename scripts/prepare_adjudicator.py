#!/usr/bin/env python3
"""prepare_adjudicator.py — Build the ADJUDICATOR file from A/B disagreements.

Implements the "double-blind, disagreements-only" handoff that the docs describe
(``docs/RATER_INSTRUCTIONS_v3.md``: "ADJUDICATOR — Only the A!=B disagreement
records (double-blind: prompt/response only)") but that the rest of the pipeline
lacked: the sealed templates for RATER_B and ADJUDICATOR are generated full-size
(147 / 63 records) by ``scripts/onboard_raters.py``, and there was no step that
surfaced ONLY the records where RATER_A and RATER_B disagree.

This script is that missing step. It runs *after* RATER_A and RATER_B are fully
annotated and *before* ADJUDICATOR starts annotating:

    1. Read RATER_A and RATER_B annotation files for one split
       (calibration or heldout).
    2. Find every record whose effective labels differ (A != B, both non-empty
       and not rater-marked uncertain — uncertain raters do not cast a valid
       vote, same rule as ``resolve_disagreements.py``).
    3. Rebuild the adjudicator file from the *sealed* template (prompt/response
       ONLY — no A/B label, no auto_label, no model/prompt ids). This preserves
       the double-blind protocol: Iqbal sees only the original material, not how
       A or B voted.
    4. Write ``{experiment_id}_ADJUDICATOR_{split}.jsonl`` containing ONLY these
       disagreement records, and record the handoff in the annotation audit log.

The output file is still in the exact sealed-template schema, so it can be fed
to the Rater Annotation tab and to ``scripts/resolve_disagreements.py`` exactly
like a full file — it is just smaller (only the truth that needs a third vote).

Usage:
    python3 scripts/prepare_adjudicator.py \
        --rater-a experiment/annotations/{exp}_RATER_A_calibration.jsonl \
        --rater-b experiment/annotations/{exp}_RATER_B_calibration.jsonl \
        --sealed experiment/sealed/templates/{exp}_calibration_template.jsonl \
        --out experiment/annotations/{exp}_ADJUDICATOR_calibration.jsonl \
        --experiment-id {exp} --split calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action  # noqa: E402

# ─────────── record key / label helpers (mirror resolve_disagreements.py) ────────
DIMENSION_LABEL_KEY = {
    "safety": "safety_label",
    "truthfulness": "truthfulness_label",
    "consistency": "consistency_label",
}


def _dimension_of(rec: dict) -> str:
    pr = rec.get("prompt_record") or {}
    return (pr.get("dimension") or "").strip()


def _label_of(rec: Optional[dict]) -> Optional[str]:
    """Extract the rater's chosen label for a record (None if absent)."""
    if rec is None:
        return None
    dim = _dimension_of(rec)
    key = DIMENSION_LABEL_KEY.get(dim)
    if not key:
        return None
    ann = rec.get("annotations") or {}
    label = ann.get(key, "")
    return label or None


def _uncertain_of(rec: Optional[dict]) -> bool:
    return bool((rec.get("flags") or {}).get("rater_uncertainty") == "uncertain")


def _record_key(rec: dict) -> Tuple[str, str]:
    """Same join key as ``resolve_disagreements.record_key``.

    Uses the opaque ``internal_key``; falls back to (dimension, prompt_id) only
    for pre-internal-key records. Consistency records are identified at GROUP
    level by their internal key (one key per multi-prompt group).
    """
    ik = rec.get("internal_key", "")
    dim = _dimension_of(rec)
    if ik:
        return (dim, ik)
    pr = rec.get("prompt_record") or {}
    pid = pr.get("prompt_id", "")
    return (dim, f"pid::{pid}")


def _load_records(path: Path) -> Dict[Tuple[str, str], dict]:
    """Load annotation records keyed by ``record_key``."""
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[_record_key(rec)] = rec
    return out


def _load_full_template(path: Path) -> List[dict]:
    """Load a full sealed template (RATER files are full-size clones of it)."""
    return [json.loads(line) for line in path.open() if line.strip()]


def _template_lookup(template_records: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Index sealed template records by record key (the ONLY source of truth).

    We always rebuild the adjudicator file from the *sealed* template, not from
    a filled rater file, so no A/B label can ever leak into what the adjudicator
    sees.
    """
    return {_record_key(r): r for r in template_records}


def _strip_annotation_fields(rec: dict) -> dict:
    """Return a deep copy of a sealed record with all rater fields blanked.

    The sealed-template schema already omits ground-truth/identity fields
    (auto_label, model_id, attack_type, prompt_id). The RATER files are clones
    of this schema with empty annotation fields, but to be extra safe we
    re-blank the annotation/flag/metadata fields here so a partially-filled
    source template can never leak a stray label into the adjudicator file.
    """
    import copy

    out = copy.deepcopy(rec)
    # Defense-in-depth: identity/ground-truth fields must be blank even if a
    # future sealed-template change ever started carrying them. The canonical
    # sealed template already strips these (see seal_experiment._template_record),
    # but the adjudicator handoff must never depend on that being true.
    pr = out.setdefault("prompt_record", {})
    for key in ("prompt_id", "model_id", "attack_type"):
        if key in pr:
            pr[key] = ""

    ann = out.setdefault("annotations", {})
    for key in ("safety_label", "truthfulness_label", "consistency_label"):
        ann[key] = ""
    ann["label_source"] = "rater_judgment"
    flags = out.setdefault("flags", {})
    flags["difficult_case"] = False
    flags["rater_uncertainty"] = ""
    flags["ai_assistance_used"] = False
    flags["ai_assistance_description"] = ""
    flags["response_invalid"] = False
    md = out.setdefault("metadata", {})
    md["notes"] = ""
    md["declared_conflicts"] = []
    return out


def _disagreement_keys(
    a: Dict[Tuple[str, str], dict],
    b: Dict[Tuple[str, str], dict],
) -> List[Tuple[str, str]]:
    """Return keys on which A and B cast *valid, different* votes.

    Mirrors the resolution-table rule from ``resolve_disagreements.py``: an
    ``uncertain`` rater does not cast a valid vote, so no adjudication is needed
    on that key (the disagreement ledger handles it separately).
    """
    keys: Set[Tuple[str, str]] = set(a) & set(b)
    result: List[Tuple[str, str]] = []
    for k in sorted(keys):
        rec_a, rec_b = a[k], b[k]
        label_a, label_b = _label_of(rec_a), _label_of(rec_b)
        # Both must have cast a real, valid (non-uncertain) vote.
        if not label_a or not label_b:
            continue
        if _uncertain_of(rec_a) or _uncertain_of(rec_b):
            continue
        if label_a != label_b:
            result.append(k)
    return result


def prepare_adjudicator(
    a_path: Path,
    b_path: Path,
    sealed_template_path: Path,
    out_path: Path,
    *,
    experiment_id: str,
    split: str,
    notes: Optional[str] = None,
) -> Dict:
    """Build the disagreements-only adjudicator file.

    Returns a report dict with the count of disagreements, the written path, and
    per-key outcome (A/B labels are included in the *report* for the researcher
    but never in the adjudicator file itself).
    """
    a = _load_records(a_path)
    b = _load_records(b_path)
    template_full = _load_full_template(sealed_template_path)
    template_index = _template_lookup(template_full)

    keys = _disagreement_keys(a, b)

    # Missing templates must fail loudly, not silently drop records.
    missing_in_template = [k for k in keys if k not in template_index]
    if missing_in_template:
        raise RuntimeError(
            "Adjudicator-prep: sealed template missing these disagreement keys: "
            f"{missing_in_template!r}. Refusing to build a silently-incomplete "
            "file (coverage invariant)."
        )

    # Rebuild from the sealed template -> double-blind.
    adjudicator_records = [
        _strip_annotation_fields(template_index[k])
        for k in keys
    ]

    # Delete any pre-existing ADJUDICATOR file FIRST, then write fresh.
    # This guarantees the adjudicator never sees a stale full-size template (or
    # a previous handoff) mixed with the new disagreements-only file: the file
    # handed to the Rater Annotation tab / resolve step is exactly what this
    # run produced, nothing more.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with out_path.open("w") as f:
        for rec in adjudicator_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # The report (researcher-side) keeps the A/B labels for traceability;
    # the *file* never contains them.
    per_key = []
    for k in keys:
        per_key.append({
            "record_key": list(k),
            "internal_key": k[1],
            "dimension": k[0],
            "label_A": _label_of(a.get(k)),
            "label_B": _label_of(b.get(k)),
            "uncertain_A": _uncertain_of(a.get(k)),
            "uncertain_B": _uncertain_of(b.get(k)),
        })

    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_id": experiment_id,
        "split": split,
        "source_a": str(a_path),
        "source_b": str(b_path),
        "n_records_a": len(a),
        "n_records_b": len(b),
        "n_disagreements_sent_to_adjudicator": len(keys),
        "n_records_adjudicator_file": len(adjudicator_records),
        "adjudicator_file": str(out_path),
        "per_key": per_key,
        "notes": notes or "",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-a", required=True, type=Path,
                        help="Filled RATER_A annotation JSONL (calibration or heldout).")
    parser.add_argument("--rater-b", required=True, type=Path,
                        help="Filled RATER_B annotation JSONL (same split).")
    parser.add_argument("--sealed", required=True, type=Path,
                        help="Sealed template JSONL for the same split "
                             "(experiment/sealed/templates/{exp}_{split}_template.jsonl).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output ADJUDICATOR JSONL path "
                             "(experiment/annotations/{exp}_ADJUDICATOR_{split}.jsonl).")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--split", required=True, choices=["calibration", "heldout"])
    parser.add_argument("--notes", default="", help="Optional researcher note.")
    args = parser.parse_args()

    report = prepare_adjudicator(
        args.rater_a,
        args.rater_b,
        args.sealed,
        args.out,
        experiment_id=args.experiment_id,
        split=args.split,
        notes=args.notes,
    )

    print(f"  {args.experiment_id} / {args.split}:")
    print(f"    RATER_A records        : {report['n_records_a']}")
    print(f"    RATER_B records        : {report['n_records_b']}")
    print(f"    A!=B disagreements      : {report['n_disagreements_sent_to_adjudicator']}")
    print(f"    ADJUDICATOR file rows  : {report['n_records_adjudicator_file']}")
    print(f"    -> {report['adjudicator_file']}")

    # Audit trail: handoff to the adjudicator must be timestamped and the file
    # hashed, so the blinding protocol is provable later.
    log_action(
        log_path=Path("experiment/logs/annotation_log.jsonl"),
        action="prepare_adjudicator",
        script="scripts/prepare_adjudicator.py",
        args={
            "--split": args.split,
            "--experiment-id": args.experiment_id,
            "--out": str(args.out),
        },
        input_paths={
            "rater_a": args.rater_a,
            "rater_b": args.rater_b,
            "sealed_template": args.sealed,
        },
        output_paths={"adjudicator": args.out},
        extra={
            "n_disagreements_sent_to_adjudicator": report["n_disagreements_sent_to_adjudicator"],
            "n_records_adjudicator_file": report["n_records_adjudicator_file"],
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

