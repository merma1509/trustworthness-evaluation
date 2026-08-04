#!/usr/bin/env python3
"""Analyze manual audit results
Computes agreement, Cohen's Kappa, confusion matrix per dimension and overall.

Usage:
    python3 scripts/analyze_audit_results.py

    # Specify custom audit file
    python3 scripts/analyze_audit_results.py \
        --input results/audit/all_audit.jsonl

    # Show full confusion matrix for each dimension
    python3 scripts/analyze_audit_results.py --verbose

Output:
    results/audit/agreement_report.json   (machine-readable)
    Console: formatted report
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agreement import compute_agreement, compute_per_dimension_agreement
from src.utils import load_jsonl, save_jsonl


def print_report(
    records: list,
    verbose: bool = False,
):
    """Print a formatted agreement report to console."""

    # ── Overall stats ────────────────────────────────────
    total = len(records)
    labelled = [r for r in records if r.get("human_label") is not None]
    n_labelled = len(labelled)
    pct_done = n_labelled / total * 100 if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"MANUAL AUDIT — AGREEMENT REPORT")
    print(f"{'=' * 70}")
    print(f"  Total records:     {total}")
    print(f"  Human-labelled:    {n_labelled}  ({pct_done:.1f}%)")
    print(f"  Pending:           {total - n_labelled}")

    if n_labelled == 0:
        print(f"\n  ⚠  No human labels found.")
        print(f"  Run annotation first, then re-run this script.")
        return

    # ── Per-dimension ────────────────────────────────────
    print(f"\n  {'─' * 70}")
    print(f"  PER-DIMENSION AGREEMENT")
    print(f"  {'─' * 70}")

    dim_agreement = compute_per_dimension_agreement(labelled)

    for dim in ["safety", "truthfulness", "consistency"]:
        if dim not in dim_agreement:
            print(f"\n  [{dim.upper()}] No data")
            continue
        da = dim_agreement[dim]
        n = da["n_valid_pairs"]
        agree = da["agreement_rate"]
        kappa = da["cohens_kappa"]

        print(f"\n  [{dim.upper()}]  n={n}")
        print(f"    Agreement rate: {agree:.1%}")
        print(f"    Cohen's Kappa:  {kappa:.4f}")

        # Interpretation
        if kappa >= 0.81:
            print(f"    Interpretation: Almost perfect agreement")
        elif kappa >= 0.61:
            print(f"    Interpretation: Substantial agreement")
        elif kappa >= 0.41:
            print(f"    Interpretation: Moderate agreement")
        elif kappa >= 0.21:
            print(f"    Interpretation: Fair agreement")
        elif kappa >= 0.0:
            print(f"    Interpretation: Slight agreement")
        else:
            print(f"    Interpretation: Poor agreement (worse than random)")

        # Per-label breakdown
        print(f"    Per-label:")
        for label, stats in sorted(da["per_label_agreement"].items()):
            print(f"      '{label}': precision={stats['precision']:.3f}, "
                  f"recall={stats['recall']:.3f}, "
                  f"f1={stats['f1']:.3f}, "
                  f"support_human={stats['human_count']}, "
                  f"support_auto={stats['auto_count']}")

        if verbose:
            print(f"    Confusion matrix (auto → human):")
            cm = da["confusion_matrix"]
            labels = sorted(set(cm.keys()) | set(
                k for v in cm.values() for k in v.keys()
            ))
            header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
            print(f"      {header}")
            for auto_label in labels:
                row = cm.get(auto_label, {})
                vals = "".join(f"{row.get(l, 0):>12}" for l in labels)
                print(f"      {auto_label:>12}{vals}")

    # ── Overall (pooled) ─────────────────────────────────
    print(f"\n  {'─' * 70}")
    print(f"  OVERALL (pooled across dimensions)")
    print(f"  {'─' * 70}")

    all_human = [r["human_label"] for r in labelled]
    all_auto = [r["auto_label"] for r in labelled]
    overall = compute_agreement(all_human, all_auto)

    print(f"    n = {overall['n_valid_pairs']}")
    print(f"    Agreement rate: {overall['agreement_rate']:.1%}")
    print(f"    Cohen's Kappa:  {overall['cohens_kappa']:.4f}")
    print(f"    Categories:     {overall['categories']}")

    if verbose or True:  # Always show overall confusion matrix
        print(f"\n    Confusion matrix (auto → human):")
        cm = overall["confusion_matrix"]
        labels = sorted(set(cm.keys()) | set(
            k for v in cm.values() for k in v.keys()
        ))
        header_label = "auto \\ human"
        header = f"{header_label:>12}" + "".join(f"{l:>12}" for l in labels)
        print(f"      {header}")
        print(f"      {'-' * (12 + 12 * len(labels))}")
        for auto_label in labels:
            row = cm.get(auto_label, {})
            vals = "".join(f"{row.get(l, 0):>12}" for l in labels)
            print(f"      {auto_label:>12}{vals}")

    # ── Error analysis ───────────────────────────────────
    print(f"\n  {'─' * 70}")
    print(f"  ERROR ANALYSIS — Disagreements")
    print(f"  {'─' * 70}")

    disagreements = [
        r for r in labelled
        if r["human_label"] != r["auto_label"]
    ]

    print(f"    Total disagreements: {len(disagreements)} "
          f"({len(disagreements)/n_labelled:.1%} of labelled)")

    if disagreements:
        # Show a few examples
        n_show = min(10, len(disagreements))
        print(f"    First {n_show} disagreements:")
        for r in disagreements[:n_show]:
            dim = r["dimension"]
            pid = r.get("prompt_id", r.get("audit_id", "?"))
            human = r["human_label"]
            auto = r["auto_label"]
            model = r.get("model", "?")
            atype = r.get("attack_type", "?")
            print(f"      [{dim}] [{model}] {pid} ({atype}): "
                  f"human={human} auto={auto}")
            # Show first 100 chars of response
            response = r.get("response", "")
            if response:
                print(f"        Response: {response[:120]}...")
            else:
                p1 = r.get("prompt_1", {})
                p2 = r.get("prompt_2", {})
                print(f"        Resp1: {p1.get('response','')[:80]}...")
                print(f"        Resp2: {p2.get('response','')[:80]}...")

    # ── Agreement by model ───────────────────────────────
    print(f"\n  {'─' * 70}")
    print(f"  AGREEMENT BY MODEL")
    print(f"  {'─' * 70}")

    by_model = defaultdict(list)
    for r in labelled:
        m = r.get("model", "unknown")
        by_model[m].append((r["human_label"], r["auto_label"]))

    for model in sorted(by_model):
        humans = [p[0] for p in by_model[model]]
        autos = [p[1] for p in by_model[model]]
        ag = compute_agreement(humans, autos)
        print(f"    {model}: n={ag['n_valid_pairs']}, "
              f"agreement={ag['agreement_rate']:.1%}, "
              f"kappa={ag['cohens_kappa']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"END OF REPORT")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze manual audit results"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="results/audit/all_audit.jsonl",
        help="Path to audit JSONL file (default: results/audit/all_audit.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/audit/agreement_report.json",
        help="Path to save agreement report JSON (default: results/audit/agreement_report.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full confusion matrices",
    )
    args = parser.parse_args()

    # Load
    if not Path(args.input).exists():
        print(f"  Input file not found: {args.input}")
        print(f"  Generate first: python3 scripts/generate_audit_samples.py")
        sys.exit(1)

    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records from {args.input}")

    # Print report
    print_report(records, verbose=args.verbose)

    # Save machine-readable report
    labelled = [r for r in records if r.get("human_label") is not None]

    if labelled:
        dim_agreement = compute_per_dimension_agreement(labelled)
        all_human = [r["human_label"] for r in labelled]
        all_auto = [r["auto_label"] for r in labelled]
        overall = compute_agreement(all_human, all_auto)

        # By model
        by_model = defaultdict(list)
        for r in labelled:
            m = r.get("model", "unknown")
            by_model[m].append((r["human_label"], r["auto_label"]))

        model_agreement = {}
        for model in sorted(by_model):
            humans = [p[0] for p in by_model[model]]
            autos = [p[1] for p in by_model[model]]
            model_agreement[model] = compute_agreement(humans, autos)

        report = {
            "pipeline": "analyze_audit_results.py",
            "total_records": len(records),
            "labelled_records": len(labelled),
            "overall": overall,
            "by_dimension": dim_agreement,
            "by_model": model_agreement,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report saved to {args.output}")

    if len(labelled) < len(records):
        remaining = len(records) - len(labelled)
        print(f"\n  ⚠  {remaining} records still pending annotation.")
        print(f"  Fill human_label field and re-run.")


if __name__ == "__main__":
    main()
