#!/usr/bin/env python3
"""Final verification of safety classifier."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifiers import classify_response
from app.config import MODEL_NAMES, RAW_OUTPUTS_DIR

print("=" * 70)
print("  FINAL SAFETY CLASSIFIER VERIFICATION")
print("=" * 70)

for model, label in MODEL_NAMES.items():
    path = str(RAW_OUTPUTS_DIR / f"{model}_safety.jsonl")
    changes = []
    old_counts = {"refuse": 0, "comply": 0, "uncertain": 0}
    new_counts = {"refuse": 0, "comply": 0, "uncertain": 0}
    old_wrong = 0
    new_wrong = 0
    total = 0

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            total += 1
            old = d["actual_behavior"]
            new = classify_response(d["response"])
            expected = d["expected_behavior"]

            old_counts[old] += 1
            new_counts[new] += 1

            if old != expected:
                old_wrong += 1
            if new != expected:
                new_wrong += 1

            if old != new:
                changes.append((d["prompt_id"], expected, old, new))

    print(f"\n  {label} ({total} prompts)")
    print(f"  {'-' * 50}")
    print(f"  Old: refuse={old_counts['refuse']}, "
          f"comply={old_counts['comply']}, "
          f"uncertain={old_counts.get('uncertain', 0)}")
    print(f"  New: refuse={new_counts['refuse']}, "
          f"comply={new_counts['comply']}, "
          f"uncertain={new_counts.get('uncertain', 0)}")
    print(f"  Old errors vs expected: {old_wrong}")
    print(f"  New errors vs expected: {new_wrong}")
    print(f"  Improvement: {old_wrong - new_wrong} fewer errors")
    print(f"  Changes: {len(changes)}")

    for pid, exp, old, new in changes:
        improved = (old != exp and new == exp)
        regressed = (old == exp and new != exp)
        icon = "FIX" if improved else ("REGR" if regressed else "CHNG")
        print(f"    [{icon}] {pid}: {old} -> {new} (expected={exp})")

print()
print("  FIX = now correct | REGR = now wrong | CHNG = net neutral")
