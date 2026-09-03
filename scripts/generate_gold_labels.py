#!/usr/bin/env python3
"""generate_gold_labels.py — Build the gold-label file

Consumes the disagreement-resolution report and emits a gold labels JSONL that
maps each scored record to its adjudicated human label plus the sealed auto label
for the later gold-vs-auto comparison

The gold file is the *single immutable* source of truth for the held-out
agreement computations. It is generated deterministically from the resolution
report + sealed ground truth — it is never hand-edited (silent-exclusion fix)

Usage:
    python3 scripts/generate_gold_labels.py \
        --resolutions experiment/agreements/{exp}_disagreements.json \
        --unseal experiment/sealed/labels/sealed_auto_labels.jsonl.enc \
        --passphrase "$SEAL_PASSPHRASE" \
        --out experiment/gold/gold_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action
from src.sealing import decrypt_json_file, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", required=True,
                        help="Disagreement-resolution report JSON (resolve_disagreements).")
    parser.add_argument("--sealed-labels", default=None,
                        help="Sealed auto labels file (.enc); optional if --auto-labels given.")
    parser.add_argument("--auto-labels", default=None,
                        help="Plain auto-labels JSON (if already unsealed).")
    parser.add_argument("--passphrase", default=None,
                        help="Passphrase to open the sealed labels file.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    res_path = Path(args.resolutions)
    if not res_path.exists():
        print(f"  ✗ Resolution report not found: {res_path}")
        return 1

    report = json.loads(res_path.read_text())

    # ── Load auto labels ──────────────────────────────────────────────────────
    if args.auto_labels:
        auto_doc = json.loads(Path(args.auto_labels).read_text())
    elif args.sealed_labels:
        if not args.passphrase:
            print("  ✗ --passphrase required to unseal auto labels.")
            return 1
        auto_doc = decrypt_json_file(Path(args.sealed_labels), args.passphrase)
    else:
        print("  ✗ Provide --auto-labels or --sealed-labels(+--passphrase).")
        return 1

    # The sealed auto payload is a list; index by the opaque internal_key so the
    # gold/auto join is exact and does not depend on prompt_id or model id.
    auto_by_key: Dict[str, dict] = {}
    for rec in auto_doc:
        key = rec.get("internal_key", "") or rec.get("prompt_id", "")
        auto_by_key[key] = rec

    # ── Emit gold labels ───────────────────────────────────────────────────────
    gold_records: List[dict] = []
    for o in report.get("outcomes", []):
        gold = o.get("gold_label")
        if gold is None:
            continue
        dim, ik = o["record_key"][0], o["record_key"][1]
        gold_records.append({
            "internal_key": ik,
            "record_key": o["record_key"],
            "dimension": dim,
            "gold_human_label": gold,
            "outcome": o["outcome"],
            "labels": {
                "RATER_A": o["label_A"],
                "RATER_B": o["label_B"],
                "ADJUDICATOR": o["label_C"],
            },
            "uncertain": o["uncertain"],
        })

    # ── Attach auto labels (for the gold-vs-auto comparison) ─────────────────
    for gr in gold_records:
        auto = auto_by_key.get(gr["internal_key"])
        gr["auto_label"] = auto["auto_label"] if auto else None
        gr["model_id"] = auto.get("model_id") if auto else None
        gr["attack_type"] = auto.get("attack_type") if auto else None
        gr["expected_behavior"] = auto.get("expected_behavior") if auto else None

    # ── Write ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for gr in gold_records:
            f.write(json.dumps(gr, ensure_ascii=False) + "\n")

    n_gold = len([g for g in gold_records if g.get("gold_human_label")])
    n_auto = len([g for g in gold_records if g.get("auto_label")])
    print(f"  Gold records written: {len(gold_records)} "
          f"(with gold label: {n_gold}, with auto label: {n_auto})")
    print(f"  -> {out_path}")

    # Audit trail ("log decryption_hash"): record the
    # gold generation together with the sealed-file hash that was opened
    sealed_for_log = Path(args.sealed_labels) if args.sealed_labels else None
    log_action(
        log_path=Path("experiment/logs/annotation_log.jsonl"),
        action="generate_gold_labels",
        script="scripts/generate_gold_labels.py",
        args={"--resolutions": args.resolutions, "--out": args.out},
        input_paths={"resolutions": res_path,
                     "sealed_labels": sealed_for_log},
        output_paths={"gold": out_path},
        extra={
            "n_records": len(gold_records),
            "n_gold": n_gold,
            "n_auto": n_auto,
            "decryption_hash": sha256_file(sealed_for_log)
            if sealed_for_log and sealed_for_log.exists() else None,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
