#!/usr/bin/env python3
"""Import human annotations from CSV back into the audit system.

Usage:
    # After filling human_label in results/annotation/{dim}_annotation.csv:
    python3 scripts/import_labels.py
    
    # Or specify files:
    python3 scripts/import_labels.py \\
        --safety results/annotation/safety_annotation.csv \\
        --truthfulness results/annotation/truthfulness_annotation.csv \\
        --consistency results/annotation/consistency_annotation.csv

Output:
    results/audit/all_audit.jsonl  (with human labels merged)
    results/disagreement_analysis.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agreement import compute_agreement, compute_per_dimension_agreement
from src.validation import compute_validation_report, compute_error_analysis
from src.utils import save_jsonl, load_jsonl
from app.config import RESULTS_DIR, MODEL_NAMES


def load_csv_labels(csv_path: Path, id_field: str = "id", label_field: str = "human_label") -> dict:
    """Load human labels from CSV. Returns {id: label}."""
    labels = {}
    if not csv_path.exists():
        print(f"  ⚠  File not found: {csv_path}")
        return labels
    
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get(label_field, "").strip().lower()
            if label:
                labels[row.get(id_field, "")] = label
    
    return labels


def load_jsonl_labels(jsonl_path: Path) -> dict:
    """Load human labels from JSONL. Returns {id: label}."""
    labels = {}
    if not jsonl_path.exists():
        return labels
    for r in load_jsonl(str(jsonl_path)):
        label = r.get("human_label", "")
        if label:
            labels[r.get("id", "")] = label
    return labels


def merge_into_audit():
    """Merge human labels from annotation files into audit records."""
    from src.audit import build_full_audit_dataset
    
    # Rebuild fresh audit from raw outputs
    model_keys = list(MODEL_NAMES.keys())
    audit = build_full_audit_dataset(
        model_labels=model_keys,
        safety_sample=18,
        truthfulness_sample=19,
        consistency_sample=8,
        random_seed=42,
    )
    
    # Load human labels from annotation files
    annotation_dir = RESULTS_DIR / "annotation"
    
    # Try CSV first, then JSONL
    safety_labels = load_csv_labels(annotation_dir / "safety_annotation.csv")
    if not safety_labels:
        safety_labels = load_jsonl_labels(annotation_dir / "safety_annotation.jsonl")
    
    truth_labels = load_csv_labels(annotation_dir / "truthfulness_annotation.csv")
    if not truth_labels:
        truth_labels = load_jsonl_labels(annotation_dir / "truthfulness_annotation.jsonl")
    
    cons_labels = load_csv_labels(annotation_dir / "consistency_annotation.csv")
    if not cons_labels:
        cons_labels = load_jsonl_labels(annotation_dir / "consistency_annotation.jsonl")
    
    all_labels = {}
    all_labels.update(safety_labels)
    all_labels.update(truth_labels)
    all_labels.update(cons_labels)
    
    print(f"  Loaded {len(all_labels)} human labels:")
    print(f"    Safety:       {len(safety_labels)}")
    print(f"    Truthfulness: {len(truth_labels)}")
    print(f"    Consistency:  {len(cons_labels)}")
    
    # Map annotation IDs → audit record IDs
    # Safety:   S_SAFE_001 → SAFE_001
    # Truthful: T_TRU_001  → TRU_001
    # Consist:  C_group_1  → group_1 (set on all records with that group_id)
    
    matched = 0
    for r in audit:
        # Try exact match
        aid = r.get("audit_id", "")
        
        # Construct expected annotation ID
        dim = r.get("dimension", "")
        if dim == "consistency":
            ann_id = f"C_{r.get('group_id', r.get('prompt_id', ''))}"
        elif dim == "safety":
            ann_id = f"S_{r.get('prompt_id', '')}"
        elif dim == "truthfulness":
            ann_id = f"T_{r.get('prompt_id', '')}"
        else:
            ann_id = ""
        
        if ann_id in all_labels:
            r["human_label"] = all_labels[ann_id]
            
            # Convert rubric labels to auto-style for agreement computation
            rl = all_labels[ann_id].lower()
            if dim == "safety":
                label_map = {"yes": "refuse", "no": "comply", "partial": "uncertain"}
                r["human_label_normalized"] = label_map.get(rl, rl)
            elif dim == "truthfulness":
                label_map = {"yes": "correct", "no": "incorrect", "refuses": "not_attempted"}
                r["human_label_normalized"] = label_map.get(rl, rl)
            elif dim == "consistency":
                label_map = {"same": "consistent", "different": "inconsistent", "similar": "consistent"}
                r["human_label_normalized"] = label_map.get(rl, rl)
            
            matched += 1
    
    print(f"  Matched {matched}/{len(all_labels)} labels to audit records")
    
    return audit


def compute_disagreement_analysis(audit_records: list) -> dict:
    """Generate disagreement analysis report."""
    labelled = [r for r in audit_records if r.get("human_label_normalized")]
    
    if not labelled:
        return {"error": "No human labels found in audit records"}
    
    # Overall agreement
    human_labels = [r["human_label_normalized"] for r in labelled]
    auto_labels = [r["human_label"] if r["human_label"] in ("refuse", "comply", "uncertain", "correct", "incorrect", "not_attempted", "consistent", "inconsistent") else r.get("auto_label", "?") for r in labelled]
    
    # Better: use auto_label field which is the normalized version
    auto_labels = []
    for r in labelled:
        dim = r.get("dimension", "")
        if dim == "consistency":
            auto_labels.append("consistent" if r.get("group_consistent", False) else "inconsistent")
        elif dim == "truthfulness":
            auto_labels.append(r.get("actual_behavior", r.get("scorer_label", "?")))
        else:
            auto_labels.append(r.get("actual_behavior", "?"))
    
    overall = compute_agreement(human_labels, auto_labels)
    
    # Per-dimension
    per_dim = compute_per_dimension_agreement(labelled)
    
    # Per-attack-type and per-model
    error_analysis = compute_error_analysis(labelled)
    
    # False positives/negatives detail
    fp = []
    fn = []
    for r in labelled:
        h = r.get("human_label_normalized", "")
        dim = r.get("dimension", "")
        if dim == "consistency":
            a = "consistent" if r.get("group_consistent", False) else "inconsistent"
        elif dim == "truthfulness":
            a = r.get("actual_behavior", "")
        else:
            a = r.get("actual_behavior", "")
        
        if h == a:
            continue
        
        # Auto too strict (auto=bad, human=good)
        human_good = h in ("refuse", "correct", "consistent", "uncertain")
        auto_good = a in ("refuse", "correct", "consistent", "uncertain")
        
        if human_good and not auto_good:
            fn.append({
                "id": r.get("audit_id", r.get("prompt_id", "")),
                "dimension": dim,
                "attack_type": r.get("attack_type", ""),
                "prompt_text": r.get("prompt_text", "")[:100],
                "response": r.get("response", "")[:200],
                "human": h,
                "auto": a,
            })
        elif not human_good and auto_good:
            fp.append({
                "id": r.get("audit_id", r.get("prompt_id", "")),
                "dimension": dim,
                "attack_type": r.get("attack_type", ""),
                "prompt_text": r.get("prompt_text", "")[:100],
                "response": r.get("response", "")[:200],
                "human": h,
                "auto": a,
            })
    
    return {
        "meta": {"total_labelled": len(labelled)},
        "overall": overall,
        "per_dimension": per_dim,
        "error_analysis": error_analysis,
        "false_positives_auto_too_optimistic": fp[:20],  # limit output size
        "false_negatives_auto_too_strict": fn[:20],
        "n_false_positives": len(fp),
        "n_false_negatives": len(fn),
    }


def main():
    parser = argparse.ArgumentParser(description="Import human labels and compute disagreement analysis")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "disagreement_analysis.json"),
                        help="Output path for disagreement analysis")
    parser.add_argument("--audit-output", type=str, default=str(RESULTS_DIR / "audit" / "all_audit.jsonl"),
                        help="Output path for audit records with merged labels")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMPORTING HUMAN LABELS")
    print("=" * 60)
    
    # Merge labels into audit records
    audit = merge_into_audit()
    
    labelled_count = sum(1 for r in audit if r.get("human_label_normalized"))
    total_count = len(audit)
    print(f"\n  Audit records: {total_count}")
    print(f"  With human labels: {labelled_count}")
    
    if labelled_count == 0:
        print("\n  ⚠  No human labels found. Annotate first:")
        print("    1. Open results/annotation/{dim}_annotation.csv")
        print("    2. Fill human_label column")
        print("    3. Re-run this script")
        return
    
    # Save merged audit
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(audit, args.audit_output)
    print(f"  Merged audit saved to {args.audit_output}")
    
    # Compute disagreement analysis
    print(f"\n  Computing disagreement analysis...")
    disagreement = compute_disagreement_analysis(audit)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(disagreement, f, indent=2, ensure_ascii=False)
    print(f"  Disagreement analysis saved to {args.output}")
    
    # Print summary
    overall = disagreement.get("overall", {})
    if overall:
        print(f"\n{'='*60}")
        print("DISAGREEMENT ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"  Overall agreement rate: {overall.get('agreement_rate', 0)*100:.1f}%")
        print(f"  Cohen's Kappa:          {overall.get('cohens_kappa', 0):.4f}")
        print(f"  N pairs:                {overall.get('n_valid_pairs', 0)}")
        print(f"  False positives (auto too optimistic): {disagreement.get('n_false_positives', 0)}")
        print(f"  False negatives (auto too strict):     {disagreement.get('n_false_negatives', 0)}")
        
        per_dim = disagreement.get("per_dimension", {})
        if per_dim:
            print(f"\n  Per dimension:")
            for dim, info in sorted(per_dim.items()):
                print(f"    {dim:<15} κ={info.get('cohens_kappa', 0):.4f}  "
                      f"agree={info.get('agreement_rate', 0)*100:.1f}%  "
                      f"n={info.get('n_valid_pairs', 0)}")
    
    print(f"\n  Done! Now run:")
    print(f"    python3 scripts/paradigm_report.py --audit {args.audit_output}")


if __name__ == "__main__":
    main()
