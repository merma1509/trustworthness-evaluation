#!/usr/bin/env python3
"""resolve_disagreements.py — Adjudicator + gold generation

Implements the three-rater design:
    * RATER_A and RATER_B annotate independently.
    * A disagreement is detected when A != B on the same record
    * ADJUDICATOR resolves those disagreements (double-blind: receives only the
      original prompt/response, not either rater's label)

Per the resolution table:

    RATER_A == RATER_B            -> label = A (=B); no adjudication needed
    RATER_A != RATER_B            -> sent to adjudicator
    ADJUDICATOR == A or == B      -> 'adjudicated agreement'
    ADJUDICATOR != A and != B     -> 'full disagreement'
    All three disagree            -> 'triple-disagreement' (kept)
    rater marked uncertain        -> not a valid label for agreement calc
    rater marked response_invalid -> excluded from scoring w/ logged reason

Usage:
    python3 scripts/resolve_disagreements.py \
        --rater-a experiment/annotations/{exp}_RATER_A_calibration.jsonl \
        --rater-b experiment/annotations/{exp}_RATER_B_calibration.jsonl \
        --adjudicator experiment/annotations/{exp}_ADJUDICATOR_calibration.jsonl \
        --split calibration \
        --out experiment/agreements/{exp}_disagreements.json \
        --experiment-id {exp}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────── Loading helpers ─────────────────────────────────────
def label_of(rec: dict) -> Optional[str]:
    """Extract the annotation label chosen by a rater for a record.

    The label lives under ``annotations.{dimension}_label`` per the sealed
    template schema. ``None`` if the rater did not choose one.
    """
    dim = rec.get("prompt_record", {}).get("dimension", "")
    key = {
        "safety": "safety_label",
        "truthfulness": "truthfulness_label",
        "consistency": "consistency_label",
    }.get(dim)
    ann = rec.get("annotations") or {}
    label = ann.get(key, "") if key else ""
    return label or None


def uncertain_of(rec: dict) -> bool:
    return (rec.get("flags") or {}).get("rater_uncertainty") == "uncertain"


def response_invalid_of(rec: dict) -> bool:
    return bool((rec.get("flags") or {}).get("response_invalid"))


def record_key(rec: dict) -> Tuple[str, str]:
    """Key identifying a unique record across raters

    Uses the opaque ``internal_key`` carried by every sealed template. This is
    the same join id embedded in the sealed ground truth (see scripts/
    seal_experiment.py), so RATER_A / RATER_B / ADJUDICATOR records for the same
    response collapse onto one key regardless of model id or prompt text. It
    reveals nothing about the answer or the model identity
    """
    pr = rec.get("prompt_record") or {}
    dim = pr.get("dimension", "") or (rec.get("prompt_record") or {}).get("dimension", "")
    # Prefer the opaque internal key; fall back to (dimension, prompt_id) only
    # if a pre-sealing record predates internal keys.
    ik = rec.get("internal_key", "")
    if ik:
        return (dim, ik)
    pid = pr.get("prompt_id", "")
    return (dim, f"pid::{pid}")


# ────────── Resolution logic ───────────────────────────────
def load_rater_annotations(path: Path) -> Dict[str, dict]:
    """Load a rater's annotations keyed by ``record_key``."""
    out = {}
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[record_key(rec)] = rec
    return out


def resolve_disagreements(
    a_files: List[Path],
    b_files: List[Path],
    adjudicator_files: List[Path],
    dimension: Optional[str] = None,
) -> Dict:
    """Compare A and B, send disagreements to the adjudicator, build gold.

    Args:
        a_files: RATER_A annotation files (calibration + heldout).
        b_files: RATER_B annotation files.
        adjudicator_files: ADJUDICATOR annotation files.
        dimension: optional dimension filter.

    Returns:
        Dict with the resolution report: per-record outcome, gold label, and
        agreement statistics.
    """
    A = {}
    for p in a_files:
        A.update(load_rater_annotations(p))
    B = {}
    for p in b_files:
        B.update(load_rater_annotations(p))
    C = {}
    for p in adjudicator_files:
        C.update(load_rater_annotations(p))

    all_keys = sorted(set(A) | set(B) | set(C))
    outcomes = []
    gold_by_key = {}
    excluded = []

    for key in all_keys:
        rec_a, rec_b, rec_c = A.get(key), B.get(key), C.get(key)

        # dimension filter
        dim = key[0]
        if dimension and dim != dimension:
            continue

        # A/B labels
        lab_a = label_of(rec_a) if rec_a else None
        lab_b = label_of(rec_b) if rec_b else None
        lab_c = label_of(rec_c) if rec_c else None

        uncertain = {
            "RATER_A": uncertain_of(rec_a) if rec_a else False,
            "RATER_B": uncertain_of(rec_b) if rec_b else False,
            "ADJUDICATOR": uncertain_of(rec_c) if rec_c else False,
        }
        resp_invalid = (
            response_invalid_of(rec_a) if rec_a else
            response_invalid_of(rec_b) if rec_b else False
        )

        # ── No-skip / no-exclusion ──
        # A malformed response is logged and EXCLUDED FROM SCORING with a reason,
        # but it is still a recorded data point — it is never silently dropped.
        if resp_invalid:
            excluded.append({
                "record_key": list(key),
                "reason": "response_invalid",
                "note": "No valid model response (malformed API output). "
                        "Kept but excluded from scoring with a logged reason.",
            })
            continue

        # ── Uncertain raters do not cast a valid vote ──
        eff_a = None if uncertain["RATER_A"] else lab_a
        eff_b = None if uncertain["RATER_B"] else lab_b
        eff_c = None if uncertain["ADJUDICATOR"] else lab_c

        # ── Resolution table ──
        if eff_a == eff_b and eff_a is not None:
            gold = eff_a
            outcome = "A=B_agreement"
        elif eff_a == eff_c and eff_a is not None and eff_b != eff_a:
            gold = eff_a
            outcome = "adjudicated_agreement_A"
        elif eff_b == eff_c and eff_b is not None and eff_a != eff_b:
            gold = eff_b
            outcome = "adjudicated_agreement_B"
        elif eff_c is not None and eff_a is not None and eff_b is not None \
                and len({eff_a, eff_b, eff_c}) == 3:
            gold = eff_c
            outcome = "triple_disagreement"
        elif eff_c is not None:
            gold = eff_c
            outcome = "full_disagreement"
        else:
            gold = None
            outcome = "unresolvable"

        gold_by_key[key] = gold
        outcomes.append({
            "record_key": list(key),
            "dimension": key[0],
            "label_A": lab_a, "label_B": lab_b, "label_C": lab_c,
            "uncertain": uncertain,
            "effective_A": eff_a, "effective_B": eff_b, "effective_C": eff_c,
            "outcome": outcome,
            "gold_label": gold,
        })

    # ── Aggregate stats ──
    from collections import Counter
    outcome_counter = Counter(o["outcome"] for o in outcomes)
    gold_dist = Counter(o["gold_label"] for o in outcomes if o["gold_label"])

    report = {
        "resolved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_recorded": len(outcomes) + len(excluded),
        "n_scored": len([o for o in outcomes if o["gold_label"] is not None]),
        "n_unresolvable": len([o for o in outcomes if o["gold_label"] is None]),
        "n_excluded_response_invalid": len(excluded),
        "outcome_counts": dict(outcome_counter),
        "gold_label_distribution": {k: v for k, v in gold_dist.items()},
        "excluded": excluded,
        "outcomes": outcomes,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-a", nargs="+", required=True,
                        help="RATER_A annotation file(s).")
    parser.add_argument("--rater-b", nargs="+", required=True,
                        help="RATER_B annotation file(s).")
    parser.add_argument("--adjudicator", nargs="+", required=True,
                        help="ADJUDICATOR annotation file(s).")
    parser.add_argument("--split", default="all",
                        choices=["calibration", "heldout", "all"])
    parser.add_argument("--dimension", default=None,
                        choices=["safety", "truthfulness", "consistency"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a_files = [Path(p) for p in args.rater_a]
    b_files = [Path(p) for p in args.rater_b]
    c_files = [Path(p) for p in args.adjudicator]

    # Union of all three is fine — resolution skips missing keys
    report = resolve_disagreements(a_files, b_files, c_files, args.dimension)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["experiment_id"] = args.experiment_id
    report["split"] = args.split
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  {args.split}: recorded={report['n_recorded']} "
          f"scored={report['n_scored']} "
          f"unresolvable={report['n_unresolvable']} "
          f"excluded_invalid={report['n_excluded_response_invalid']}")
    print(f"  outcomes: {report['outcome_counts']}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
