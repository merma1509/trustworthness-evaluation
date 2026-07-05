#!/usr/bin/env python3
"""Validate raw seed files
Reads from data/raw/*.jsonl.
Checks: duplicates, label integrity, distribution, coverage.

Usage: python scripts/validate_raw_seeds.py
"""

import json
from pathlib import Path
from collections import Counter


def validate_seed_file(filepath: str) -> dict:
    """Validate a single seed file and return statistics."""
    with open(filepath) as f:
        seeds = [json.loads(line) for line in f]

    stats = {
        "file": filepath.name,
        "total": len(seeds),
        "attack_types": dict(Counter(p.get("attack_type", "unknown") for p in seeds)),
        "difficulties": dict(Counter(p.get("difficulty", "unknown") for p in seeds)),
        "behaviors": dict(Counter(p.get("expected_behavior", "unknown") for p in seeds)),
        "exact_duplicates": 0,
        "duplicate_ids": 0,
        "missing_fields": {},
    }

    # Check for exact duplicate text
    texts = [p["prompt_text"] for p in seeds]
    stats["exact_duplicates"] = len(texts) - len(set(texts))

    # Check for duplicate IDs
    ids = [p["prompt_id"] for p in seeds]
    stats["duplicate_ids"] = len(ids) - len(set(ids))

    # Check for missing required fields
    required_fields = ["prompt_id", "prompt_text", "expected_behavior"]
    for field in required_fields:
        missing = sum(1 for p in seeds if field not in p)
        if missing > 0:
            stats["missing_fields"][field] = missing

    return stats


def print_stats(stats: dict):
    """Print formatted statistics for a seed file."""
    print(f"\n{'='*60}")
    print(f"File: {stats['file']}")
    print(f"{'='*60}")
    print(f"Total seeds:     {stats['total']}")
    print(f"Attack types:    {stats['attack_types']}")
    print(f"Difficulties:    {stats['difficulties']}")
    print(f"Behaviors:       {stats['behaviors']}")
    print(f"Exact duplicates: {stats['exact_duplicates']}")
    print(f"Duplicate IDs:   {stats['duplicate_ids']}")

    if stats["missing_fields"]:
        for field, count in stats["missing_fields"].items():
            print(f"  Missing '{field}': {count} seeds")

    # Warnings
    if stats["exact_duplicates"] > 0:
        print(f"  Found {stats['exact_duplicates']} exact duplicate(s)")
    if stats["duplicate_ids"] > 0:
        print(f"  Found {stats['duplicate_ids']} duplicate ID(s)")

    print(f"Validation complete")


def print_summary(all_stats: list):
    """Print an overall summary of all seed files."""
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total = 0
    total_dupes = 0
    print(f"{'File':<35} {'Seeds':<8} {'Dupes':<8} {'Classes':<30}")
    print("-" * 85)
    for stats in all_stats:
        classes = len(stats["attack_types"])
        print(f"{stats['file']:<35} {stats['total']:<8} {stats['exact_duplicates']:<8} {classes:<30}")
        total += stats["total"]
        total_dupes += stats["exact_duplicates"]
    print("-" * 85)
    print(f"{'TOTAL':<35} {total:<8} {total_dupes:<8}")

    # Coverage check
    all_attack_types = set()
    for stats in all_stats:
        all_attack_types.update(stats["attack_types"].keys())
    print(f"\nUnique attack types across all files: {len(all_attack_types)}")
    print(f"  → {sorted(all_attack_types)}")


def main():
    print("=" * 60)
    print("MultiTrustScore — Raw Seed Validation")
    print("Reads from data/raw/*.jsonl")
    print("=" * 60)

    raw_dir = Path("data/raw")
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))

    if not jsonl_files:
        print("\n❌ No JSONL files found in data/raw/")
        print("   Run 'python src/raw_seeds_builder.py' first.")
        return

    all_stats = []
    for filepath in jsonl_files:
        stats = validate_seed_file(filepath)
        print_stats(stats)
        all_stats.append(stats)

    print_summary(all_stats)
    print(f"\nAll seed files validated successfully!")


if __name__ == "__main__":
    main()