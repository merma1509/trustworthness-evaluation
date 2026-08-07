#!/usr/bin/env python3
"""Test and compare old vs new truthfulness classifier."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifiers import classify_truthfulness
from app.config import MODEL_NAMES, RAW_OUTPUTS_DIR


def main():
    print("=" * 70)
    print("  TRUTHFULNESS CLASSIFIER COMPARISON")
    print("=" * 70)

    for model, label in MODEL_NAMES.items():
        path = str(RAW_OUTPUTS_DIR / f"{model}_truthfulness.jsonl")
        changes = []
        old_correct = 0
        new_correct = 0
        total = 0

        with open(path) as f:
            for line in f:
                d = json.loads(line)
                total += 1
                old = d["actual_behavior"]
                new = classify_truthfulness(d["response"], d.get("attack_type", ""))
                expected = d["expected_behavior"]

                if old == expected:
                    old_correct += 1
                if new == expected:
                    new_correct += 1

                if old != new:
                    changes.append((d["prompt_id"], d.get("attack_type", "?"), expected, old, new, d["response"][:100]))

        print(f"\n  {label} ({total} prompts)")
        print(f"  {'-' * 50}")
        print(f"  Old accuracy vs expected: {old_correct}/{total} = {old_correct/total*100:.1f}%")
        print(f"  New accuracy vs expected: {new_correct}/{total} = {new_correct/total*100:.1f}%")
        print(f"  Changes: {len(changes)}")

        for pid, atype, exp, old, new, preview in changes:
            improved = (old != exp and new == exp)
            regressed = (old == exp and new != exp)
            icon = "FIX" if improved else ("REGR" if regressed else "CHNG")
            print(f"    [{icon}] {pid} ({atype}): {old} -> {new} (expected={exp})")
            if regressed:
                print(f"           \"{preview}...\"")

    print()
    print("  NOTE: 'Regressions' may be legitimate — expected_behavior is normative,")
    print("  the classifier may correctly identify model's actual behavior.")
    print()


if __name__ == "__main__":
    main()
