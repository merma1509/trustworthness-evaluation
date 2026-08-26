#!/usr/bin/env python3
"""backfill_audit.py

Backfill human gold labels into ``all_audit_full.jsonl`` so downstream
scripts (``paradigm_report.py``, dashboard, etc.) can read human labels
directly from the audit file without needing a separate lookup step.

Reads from three sources (all optional):
  1. Experiment held-out report  → adjudicated gold labels (majority vote)
  2. Calibration annotation files → agreed labels (where ann1 == ann2)
  3. Ground-truth mapping         → bridges anon_id in reports → real audit_id

Usage:
    python3 scripts/backfill_audit.py \
        --audit experiment/all_audit_full.jsonl \
        --ground-truth experiment/blinded/ground_truth_blinded.json \
        --experiment-report experiment/held_out_agreement_report.json \
        --calibration-ann experiment/held_out_work/ann1_calibration.jsonl \
        --calibration-ann experiment/held_out_work/ann2_calibration.jsonl \
        --output experiment/all_audit_full.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _clean(value):
    """Normalise a raw annotator value or return ``None`` if unset."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "null", "-", "--", "skip"):
        return None
    return s


def _load_ground_truth(path: Path) -> dict:
    """Load the analyst-only ground-truth mapping."""
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        doc = json.load(f)
    return doc


def _build_backfill_map(args) -> dict:
    """Build {audit_id: human_label} for every record with an adjudicated gold."""
    backfill = {}

    # ── 1. Ground-truth mapping ──────────────────────────────────────────────
    gt = _load_ground_truth(Path(args.ground_truth)) if args.ground_truth else {}
    by_anon = gt.get("by_anon_id", {})

    # anon_id -> [real audit_ids]
    anon_to_audit = defaultdict(list)
    for anon_id, entries in by_anon.items():
        for e in entries:
            anon_to_audit[anon_id].append(e.get("audit_id"))

    # ── 2. Held-out gold labels from experiment report ───────────────────────
    if args.experiment_report and Path(args.experiment_report).exists():
        with Path(args.experiment_report).open() as f:
            report = json.load(f)

        gold_records = report.get("overall", {}).get("adjudicated", {}).get("records", [])
        for rec in gold_records:
            anon_id = rec.get("audit_id")   # 'audit_id' field actually holds anon_id
            label = rec.get("human_label")
            if not anon_id or not label:
                continue
            for real_aid in anon_to_audit.get(anon_id, []):
                backfill[real_aid] = label
        print(f"  From experiment report: {len(gold_records)} adjudicated records, "
              f"{len(backfill)} audit_ids mapped")

    # ── 3. Calibration agreed labels ─────────────────────────────────────────
    if args.calibration_ann:
        ann_files = [Path(p) for p in args.calibration_ann]
        existing = [f for f in ann_files if f.exists()]

        if existing:
            # Load all annotator labels per anon_id
            cal_by_ann = defaultdict(dict)
            for f in existing:
                name = f.stem
                for line in f.open():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    aid = rec.get("anon_id") or rec.get("audit_id")
                    if not aid:
                        continue
                    label = _clean(rec.get("human_label"))
                    if label:
                        cal_by_ann[aid][name] = label

            cal_gold = {}
            for anon_id, labels in cal_by_ann.items():
                values = list(labels.values())
                if len(values) >= 2 and values[0] == values[1]:
                    cal_gold[anon_id] = values[0]  # agreed, take the shared label
                # else: tie, no adjudicated gold for calibration

            for anon_id, label in cal_gold.items():
                for real_aid in anon_to_audit.get(anon_id, []):
                    if real_aid not in backfill:  # held-out labels take priority
                        backfill[real_aid] = label

            print(f"  From calibration: {len(cal_gold)} agreed labels added "
                  f"({len(backfill)} total audit_ids now mapped)")
        else:
            print(f"  From calibration: no calibration files found "
                  f"(expected: {', '.join(str(f) for f in ann_files)}). "
                  f"Skipping calibration backfill.")
            print(f"    → {len(backfill)} audit_ids mapped from experiment report only")

    return backfill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True,
                        help="Input audit JSONL (all_audit_full.jsonl)")
    parser.add_argument("--ground-truth",
                        help="Path to ground_truth_blinded.json (bridges anon_id → audit_id)")
    parser.add_argument("--experiment-report",
                        help="Path to experiment held-out agreement report (.json)")
    parser.add_argument("--calibration-ann", action="append", default=[],
                        dest="calibration_ann",
                        help="Calibration annotation JSONL (can be repeated). "
                             "Gitignored — only needed when calibration files are present.")
    parser.add_argument("--output", required=True,
                        help="Output path (can be same as --audit for in-place update)")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"✗ Audit file not found: {args.audit}")
        return 1

    # Build the backfill map
    print("Building backfill map...")
    backfill = _build_backfill_map(args)
    print(f"  Total audit_ids to backfill: {len(backfill)}")

    if not backfill:
        print("✗ No human labels to backfill. Check that experiment report and/or "
              "calibration files exist.")
        return 1

    # Load and update audit records
    records = []
    with audit_path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    updated = 0
    for rec in records:
        aid = rec.get("audit_id")
        if aid in backfill:
            rec["human_label"] = backfill[aid]
            updated += 1

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Report
    n_filled = sum(1 for r in records if r.get("human_label"))
    print(f"\n{'='*60}")
    print("Backfill complete")
    print(f"  Output: {out_path}")
    print(f"  Total records: {len(records)}")
    print(f"  With human_label: {n_filled} ({n_filled/len(records)*100:.1f}%)")
    print(f"  Null: {len(records) - n_filled}")

    dims = defaultdict(lambda: {"total": 0, "filled": 0})
    for r in records:
        d = r.get("dimension", "unknown")
        dims[d]["total"] += 1
        if r.get("human_label"):
            dims[d]["filled"] += 1

    print("\nPer dimension:")
    for dim in ["safety", "truthfulness", "consistency"]:
        c = dims.get(dim, {"total": 0, "filled": 0})
        print(f"  {dim:<14}: {c['filled']:3d}/{c['total']:3d} "
              f"({c['filled']/c['total']*100 if c['total'] else 0:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
