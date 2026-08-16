#!/usr/bin/env python3
"""Rebuild the audit JSONL with ``attack_type`` carried through.

RQ2 needs a real attack-type breakdown. The existing ``all_audit.jsonl`` was
written with only ``{audit_id, dimension, model, auto_label, human_label}`` and
dropped ``attack_type``. This script re-joins each audit record to the raw
per-prompt outputs (by prompt_id / group_id) and enriches it with
``attack_type`` (+ provenance details already present in the raw rows).

Usage:
    python3 scripts/rebuild_audit_with_attack_type.py \
        --audit results/audit/all_audit.jsonl \
        --output results/audit/all_audit.jsonl

Output:
    A JSONL where every record has a non-empty ``attack_type`` and no
    ``attack_type`` is ``unknown`` unless the underlying raw row was ``unknown``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import save_jsonl

# model labels as they appear in audit_id prefixes
MODEL_TOKENS = ["gemma3_4b", "llama3_1_8b", "llama3.1_8b"]
# dimension tokens that split the audit_id
DIM_TOKENS = {
    "safety": "safety",
    "truth": "truthfulness",
    "cons": "consistency",
}


def _load_raw_lookup(raw_outputs_dir: Path) -> dict:
    """Build {model: {dimension: {prompt_id: record}}} from raw outputs."""
    lookup = {}
    for path in raw_outputs_dir.glob("*.jsonl"):
        if not path.name.endswith(".jsonl"):
            continue
        # parse model_dimension.jsonl
        stem = path.stem  # e.g. gemma3_4b_safety
        tokens = stem.split("_")
        # model is the first two tokens joined
        model = "_".join(tokens[:2])
        dim = tokens[2] if len(tokens) > 2 else "unknown"
        if model not in lookup:
            lookup[model] = {}
        lookup[model][dim] = {}
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lookup[model][dim][rec.get("prompt_id")] = rec
    return lookup


def _extract_prompt_id(audit_id: str) -> str:
    """Extract the prompt_id (or group id) from ``model_dim_<ID>``."""
    for mtok in MODEL_TOKENS:
        if audit_id.startswith(mtok + "_"):
            rest = audit_id[len(mtok) + 1:]
            break
    else:
        return audit_id
    for _dim_tok, dim_name in DIM_TOKENS.items():
        prefix = _dim_tok + "_"
        if rest.startswith(prefix):
            return rest[len(prefix):]
    return rest


def enrich_record(rec: dict, lookup: dict) -> dict:
    """Return a copy of ``rec`` enriched with ``attack_type`` (and details)."""
    out = dict(rec)
    model = rec.get("model")
    dim = rec.get("dimension")
    attr = _extract_prompt_id(rec.get("audit_id", ""))

    if dim == "consistency":
        # attr is the group id (e.g. 'group_7'); the raw output rows share
        # group_id. Find any row in the consistency file for that group.
        rows = lookup.get(model, {}).get("consistency", {})
        for r in rows.values():
            if str(r.get("group_id")) == str(attr):
                out["attack_type"] = r.get("attack_type", "unknown")
                out["group_id"] = r.get("group_id", attr)
                break
    else:
        rows = lookup.get(model, {}).get(dim, {})
        r = rows.get(attr)
        if r is not None:
            out["attack_type"] = r.get("attack_type", "unknown")

    # Fall back to a safe default only if still missing.
    out.setdefault("attack_type", "unknown")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", "-a", default="results/audit/all_audit.jsonl")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--raw", default="results/raw_outputs")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    raw_dir = Path(args.raw)
    out_path = Path(args.output) if args.output else audit_path

    if not audit_path.exists():
        print(f"Audit file not found: {audit_path}")
        sys.exit(1)

    lookup = _load_raw_lookup(raw_dir)

    records = []
    with audit_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            enriched = enrich_record(rec, lookup)
            records.append(enriched)

    save_jsonl(records, str(out_path))
    print(f"Wrote {len(records)} enriched records to {out_path}")

    missing = sum(
        1 for r in records
        if not r.get("attack_type") or r.get("attack_type") == "unknown"
    )
    print(
        f"Records with missing/unknown attack_type after enrichment: "
        f"{missing}/{len(records)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

