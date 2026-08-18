#!/usr/bin/env python3
"""Generate a blinded, multi-rater annotation dataset.

The audit leaks ``auto_label`` and similarity fields
to the annotator, and that there was no calibration / held-out split. This
script builds an *annotatable* blinded JSONL (prompt + response only, no
``auto_label``, no similarity) with a calibration / held-out split made at the
**unique prompt/group unit** level.

The audit file (``all_audit.jsonl``) may hold only label metadata; the actual
prompt/response text is joined from ``results/raw_outputs/*.jsonl``.

Usage:
    python3 scripts/generate_blinded_annotation.py \
        --audit results/audit/all_audit.jsonl \
        --output results/audit/blinded \
        --calibration-ratio 0.7
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MODEL_TOKENS = ["gemma3_4b", "llama3_1_8b", "llama3.1_8b"]


def _parse_audit_id(audit_id: str) -> tuple:
    """Return (model, dimension, key) from e.g. ``gemma3_4b_truth_BEN_003``."""
    for mtok in MODEL_TOKENS:
        if audit_id.startswith(mtok + "_"):
            rest = audit_id[len(mtok) + 1:]
            for tok, dim in [("safety", "safety"), ("truth", "truthfulness"),
                             ("cons", "consistency")]:
                if rest.startswith(tok + "_"):
                    return mtok, dim, rest[len(tok) + 1:]
    return None, None, None


def _load_raw_lookup(raw_dir: Path) -> dict:
    lookup = {}
    for path in raw_dir.glob("*.jsonl"):
        tokens = path.stem.split("_")
        model = "_".join(tokens[:2])
        dim = tokens[2] if len(tokens) > 2 else "unknown"
        lookup.setdefault(model, {}).setdefault(dim, {})
        for line in path.open():
            rec = json.loads(line)
            for key in ("prompt_id", "group_id"):
                if rec.get(key):
                    lookup[model][dim][rec[key]] = rec
    return lookup


def _blinded_record(rec: dict, raw: dict) -> dict:
    model, dim, key = _parse_audit_id(rec.get("audit_id", ""))
    out = {
        "audit_id": rec.get("audit_id"),
        "dimension": rec.get("dimension"),
    }
    source = raw.get(model, {}).get(dim, {})
    if dim == "consistency":
        gid = rec.get("group_id", key)

        # The raw file stores one row per PROMPT within a group, not an
        # aggregated group object. Collect every row whose group_id matches
        # and present the full set of (prompt, response) pairs so that an
        # annotator can actually judge consistency between responses.
        group_rows = [v for v in source.values() if v.get("group_id") == gid]
        out["group_id"] = gid

        out["attack_type"] = rec.get("attack_type") or (
            group_rows[0].get("attack_type") if group_rows else ""
        )
        out["pairs"] = [
            {
                "prompt_id": r.get("prompt_id", ""),
                "prompt": r.get("prompt_text", ""),
                "response": r.get("response", ""),
            }
            for r in group_rows
        ]
        # Backward-compatible convenience fields too.
        out["responses"] = [r.get("response", "") for r in group_rows]
        out["prompts"] = [r.get("prompt_text", "") for r in group_rows]
    else:
        row = source.get(rec.get("prompt_id") or key, {})
        out["prompt_id"] = rec.get("prompt_id") or key
        out["prompt"] = row.get("prompt_text", rec.get("prompt_text"))
        out["response"] = row.get("response", rec.get("response"))
    # No auto_label, no similarity fields are emitted here.
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="results/audit/all_audit.jsonl")
    parser.add_argument("--raw", default="results/raw_outputs")
    parser.add_argument("--output", default="results/audit/blinded")
    parser.add_argument("--calibration-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit_path = Path(args.audit)
    raw_dir = Path(args.raw)
    out_dir = Path(args.output)
    if not audit_path.exists():
        print(f"Audit file not found: {audit_path}")
        sys.exit(1)

    records = [json.loads(line) for line in audit_path.open() if line.strip()]
    raw = _load_raw_lookup(raw_dir)

    # Blinded records first, then split at unit level.
    blinded = [_blinded_record(r, raw) for r in records]

    def unit_key(b):
        return b.get("group_id") or b.get("prompt_id") or b.get("audit_id")

    by_unit = defaultdict(list)
    for b in blinded:
        by_unit[unit_key(b)].append(b)
    units = sorted(by_unit.keys())

    rng = random.Random(args.seed)
    rng.shuffle(units)
    n_cal = max(1, round(len(units) * args.calibration_ratio))
    cal_units, hold_units = set(units[:n_cal]), set(units[n_cal:])

    cal_recs = [r for u in cal_units for r in by_unit[u]]
    hold_recs = [r for u in hold_units for r in by_unit[u]]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "blinded_annotation_calibration.jsonl").open("w") as f:
        for r in cal_recs:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "blinded_annotation_heldout.jsonl").open("w") as f:
        for r in hold_recs:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "total_records": len(records),
        "total_units": len(units),
        "calibration_records": len(cal_recs),
        "heldout_records": len(hold_recs),
        "calibration_ratio": args.calibration_ratio,
        "note": "Blinded: prompt+response only; auto_label & similarity REMOVED.",
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Wrote {len(cal_recs)} calibration + {len(hold_recs)} "
        f"held-out blinded records to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
