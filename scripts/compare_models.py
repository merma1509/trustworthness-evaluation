#!/usr/bin/env python3
"""Model comparison using paired bootstrap tests
Compares two models across all dimensions with proper paired statistics.

Usage:
    python3 scripts/compare_models.py \
        --model1 results/raw_outputs/gemma3_4b_%%s.jsonl \
        --model2 results/raw_outputs/llama3.1_8b_%%s.jsonl \
        --label1 gemma3:4b --label2 llama3.1:8b \
        --output results/model_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import (
    compute_paired_difference_ci,
    compute_clustered_consistency_ci,
)

DIMENSIONS = ["safety", "truthfulness", "consistency"]


def load_raw_outputs(filepath: str) -> List[Dict]:
    """Load raw outputs from a JSONL file"""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_per_prompt_scores(
    records: List[Dict], dimension: str
) -> List[float]:
    """Extract per-prompt 0/1 correctness scores.

    For safety / truthfulness, each record has 'is_correct'.

    Returns:
        List of floats: 1.0 if correct, 0.0 otherwise.
    """
    scores = []
    for r in records:
        is_correct = r.get("is_correct", False)
        scores.append(1.0 if is_correct else 0.0)
    return scores


def extract_group_consistency(
    records: List[Dict],
) -> Dict[str, bool]:
    """Extract per-group consistency booleans.

    Groups records by group_id, returns {group_id: bool}
    Only includes multi-prompt groups (singletons excluded).

    Returns:
        Dict mapping group_id -> bool (consistent or not).
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in records:
        gid = r.get("group_id", r.get("prompt_id", "unknown"))
        groups[gid].append(r)

    result = {}
    for gid, recs in groups.items():
        if len(recs) == 1:
            continue  # Skip singletons
        # Get the 'group_consistent' flag — all records in a group share it
        group_consistent = any(r.get("group_consistent", False) for r in recs)
        result[gid] = group_consistent

    return result


def compare_dimension(
    dim_name: str,
    model1_records: List[Dict],
    model2_records: List[Dict],
) -> Dict:
    """Compare two models on a single dimension.

    For safety / truthfulness: uses compute_paired_difference_ci()
    For consistency:         uses compute_clustered_consistency_ci()
    """
    if dim_name in ("safety", "truthfulness"):
        m1_scores = extract_per_prompt_scores(model1_records, dim_name)
        m2_scores = extract_per_prompt_scores(model2_records, dim_name)

        result = compute_paired_difference_ci(m1_scores, m2_scores)

    elif dim_name == "consistency":
        m1_groups = extract_group_consistency(model1_records)
        m2_groups = extract_group_consistency(model2_records)

        result = compute_clustered_consistency_ci(m1_groups, m2_groups)

    else:
        raise ValueError(f"Unknown dimension: {dim_name}")

    # Add raw scores for display
    if dim_name in ("safety", "truthfulness"):
        m1_pct = sum(m1_scores) / len(m1_scores) * 100 if m1_scores else 0
        m2_pct = sum(m2_scores) / len(m2_scores) * 100 if m2_scores else 0
    else:
        m1_pct = sum(1 for v in m1_groups.values() if v) / len(m1_groups) * 100 if m1_groups else 0
        m2_pct = sum(1 for v in m2_groups.values() if v) / len(m2_groups) * 100 if m2_groups else 0

    result["model1_score_pct"] = round(m1_pct, 1)
    result["model2_score_pct"] = round(m2_pct, 1)
    result["dimension"] = dim_name

    return result


def interpret_result(result: Dict, label1: str, label2: str) -> str:
    """Human-readable summary of a comparison result."""
    diff = result["mean_difference"]
    ci = (result["ci_lower"], result["ci_upper"])
    p = result["p_value"]
    m1 = result["model1_score_pct"]
    m2 = result["model2_score_pct"]

    n = result.get("n_pairs") or result.get("n_groups", 0)

    lines = []
    lines.append(f"  Scores: {label1}={m1}%  vs  {label2}={m2}%  (n={n})")
    lines.append(f"  Difference: {diff:+.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  p={p:.4f}")

    if p < 0.05 and diff > 0:
        lines.append(f"  ➜ {label1} is significantly *better* than {label2} (p={p:.4f})")
    elif p < 0.05 and diff < 0:
        lines.append(f"  ➜ {label2} is significantly *better* than {label1} (p={p:.4f})")
    elif p >= 0.05:
        lines.append(f"  ➜ No significant difference (p={p:.4f})")
        # Check if we have enough power
        if p > 0.20:
            lines.append(f"    (Note: small sample — limited power to detect differences)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Model comparison with paired bootstrap tests"
    )
    parser.add_argument(
        "--model1", "-m1", type=str, required=True,
        help="Path template for model 1 raw outputs. Use %%s for dimension "
             "(e.g. results/raw_outputs/gemma3_4b_%%s.jsonl)"
    )
    parser.add_argument(
        "--model2", "-m2", type=str, required=True,
        help="Path template for model 2 raw outputs. Use %%s for dimension."
    )
    parser.add_argument(
        "--label1", "-l1", type=str, default="Model 1",
        help="Display label for model 1"
    )
    parser.add_argument(
        "--label2", "-l2", type=str, default="Model 2",
        help="Display label for model 2"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="results/model_comparison.json",
        help="Output path for comparison results"
    )
    parser.add_argument(
        "--dimensions", "-d", type=str, nargs="+",
        default=DIMENSIONS,
        help=f"Dimensions to compare (default: {' '.join(DIMENSIONS)})"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MODEL COMPARISON — Paired Bootstrap Tests")
    print("=" * 70)
    print(f"{args.label1}  vs  {args.label2}")
    print()

    all_results = {}

    for dim in args.dimensions:
        # Load raw outputs
        m1_path = args.model1.replace("%s", dim)
        m2_path = args.model2.replace("%s", dim)

        if not Path(m1_path).exists():
            print(f"  [SKIP] {dim}: {m1_path} not found")
            continue
        if not Path(m2_path).exists():
            print(f"  [SKIP] {dim}: {m2_path} not found")
            continue

        m1_records = load_raw_outputs(m1_path)
        m2_records = load_raw_outputs(m2_path)

        if not m1_records or not m2_records:
            print(f"  [SKIP] {dim}: empty records")
            continue

        print(f"  ───── {dim.upper()} ─────")
        result = compare_dimension(dim, m1_records, m2_records)
        all_results[dim] = result
        print(interpret_result(result, args.label1, args.label2))
        print()

    # Save
    output = {
        "comparison": {
            "model1": {"label": args.label1, "path_template": args.model1},
            "model2": {"label": args.label2, "path_template": args.model2},
        },
        "dimensions": all_results,
        "note": (
            "Paired bootstrap CIs. For safety / truthfulness: pairwise "
            "prompt-level.  For consistency: clustered (group-level).  "
            "p-value is one-sided for H0: diff ≤ 0 (model1 not better)."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
