#!/usr/bin/env python3
"""Generate stratified audit samples for human annotation

Usage:
    # All models + dimensions (50% sample)
    python3 scripts/generate_audit_samples.py --sample-ratio 0.5

    # All models + dimensions (all data, no sampling)
    python3 scripts/generate_audit_samples.py

    # Single model + single dimension
    python3 scripts/generate_audit_samples.py \
        --models gemma3_4b --dimensions consistency

    # Separate files per dimension
    python3 scripts/generate_audit_samples.py --split-by-dimension

Output:
    results/audit/all_audit.jsonl       (or dimension-specific files)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit import build_full_audit_dataset
from src.utils import save_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified audit samples for human annotation",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=["gemma3_4b", "llama3.1_8b"],
        help="Model labels to include (default: gemma3_4b llama3.1_8b)",
    )
    parser.add_argument(
        "--dimensions", "-d",
        type=str,
        nargs="+",
        choices=["safety", "truthfulness", "consistency"],
        default=["safety", "truthfulness", "consistency"],
        help="Dimensions to include (default: all)",
    )
    parser.add_argument(
        "--sample-ratio", "-r",
        type=float,
        default=None,
        help="Fraction of each stratum to sample (e.g. 0.5 = 50%%)",
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=None,
        help="Max records per stratum (overrides --sample-ratio)",
    )
    parser.add_argument(
        "--split-by-dimension",
        action="store_true",
        help="Output separate files per dimension (default: merged)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/audit",
        help="Output directory (default: results/audit)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max consistency pairs per model (to avoid explosion)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Map dimension to sample sizes
    safety_sample = truthfulness_sample = consistency_sample = None
    model_labels = args.models

    if args.sample_ratio is not None:
        # Estimate from raw output counts
        from src.audit import load_model_outputs
        total_safety = total_truth = total_cons = 0
        for ml in model_labels:
            outputs = load_model_outputs(ml)
            total_safety += len(outputs.get("safety", []))
            total_truth += len(outputs.get("truthfulness", []))
            # Consistency pairs: sum of n*(n-1)/2 per group
            from collections import defaultdict
            for dim, records in outputs.items():
                if dim == "consistency":
                    groups = defaultdict(list)
                    for r in records:
                        gid = r.get("group_id", r.get("prompt_id", "unknown"))
                        groups[gid].append(r)
                    for grp in groups.values():
                        if len(grp) >= 2:
                            total_cons += len(grp) * (len(grp) - 1) // 2

        safety_sample = max(1, int(total_safety * args.sample_ratio)) if "safety" in args.dimensions else None
        truthfulness_sample = max(1, int(total_truth * args.sample_ratio)) if "truthfulness" in args.dimensions else None
        consistency_sample = max(1, int(total_cons * args.sample_ratio)) if "consistency" in args.dimensions else None
        if consistency_sample is not None and args.max_pairs is not None:
            consistency_sample = min(consistency_sample, args.max_pairs)

        print(f"  Sampling: safety={safety_sample}, truthfulness={truthfulness_sample}, "
              f"consistency={consistency_sample}")

    elif args.sample_size is not None:
        if "safety" in args.dimensions:
            safety_sample = args.sample_size
        if "truthfulness" in args.dimensions:
            truthfulness_sample = args.sample_size
        if "consistency" in args.dimensions:
            consistency_sample = args.sample_size
            if args.max_pairs is not None:
                consistency_sample = min(consistency_sample, args.max_pairs)

    print("=" * 60)
    print("MANUAL AUDIT — Generate Samples")
    print("=" * 60)
    print(f"  Models:     {model_labels}")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Output:     {args.output_dir}/")
    print()

    # Build the full audit dataset
    all_audit = build_full_audit_dataset(
        model_labels=model_labels,
        safety_sample=safety_sample if "safety" in args.dimensions else None,
        truthfulness_sample=truthfulness_sample if "truthfulness" in args.dimensions else None,
        consistency_sample=consistency_sample if "consistency" in args.dimensions else None,
        random_seed=args.seed,
    )

    print(f"\n  Total audit records: {len(all_audit)}")

    # Summary per dimension
    dim_counts = {}
    for r in all_audit:
        dim = r["dimension"]
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
    for dim in sorted(dim_counts):
        print(f"    {dim}: {dim_counts[dim]}")

    # Summary per model
    model_counts = {}
    for r in all_audit:
        m = r["model"]
        model_counts[m] = model_counts.get(m, 0) + 1
    for m in sorted(model_counts):
        print(f"    {m}: {model_counts[m]}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_by_dimension:
        for dim in args.dimensions:
            dim_records = [r for r in all_audit if r["dimension"] == dim]
            out_path = out_dir / f"audit_{dim}.jsonl"
            save_jsonl(dim_records, str(out_path))
            print(f"  Saved {len(dim_records)} records to {out_path}")

        # Also save rubric reference
        rubric_path = "MANUAL_AUDIT_RUBRIC.md"
        if Path(rubric_path).exists():
            import shutil
            shutil.copy(rubric_path, str(out_dir / "RUBRIC.md"))
            print(f"  Copied rubric to {out_dir / 'RUBRIC.md'}")
    else:
        out_path = out_dir / "all_audit.jsonl"
        save_jsonl(all_audit, str(out_path))
        print(f"  Saved {len(all_audit)} records to {out_path}")

    print()
    print("  NEXT STEP:")
    print(f"    1. Open {out_path}")
    print("    2. Fill in 'human_label' for each entry")
    print("    3. Run: python3 scripts/analyze_audit_results.py")
    print()


if __name__ == "__main__":
    main()
