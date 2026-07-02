#!/usr/bin/env python3
"""build_datasets.py
Run the full dataset generation pipeline.
Usage: python scripts/build_datasets.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# from src.dataset_builder import DatasetBuilder

from src.raw_seeds_builder import DatasetBuilder

if __name__ == "__main__":
    print("=" * 60)
    print("MultiTrustScore — Dataset Builder")
    print("=" * 60)

    builder = DatasetBuilder(seed=42)

    dimensions = {
        "safety": (builder.get_safety_seeds(), 30, 10),
        "robustness": (builder.get_robustness_seeds(), 20, 10),
        "consistency": (builder.get_consistency_seeds(), 10, 5),
        "hallucination": (builder.get_hallucination_seeds(), 20, 10),
        "uncertainty": (builder.get_uncertainty_seeds(), 20, 10),
    }

    benign = builder.get_benign_controls()

    print(f"\nBuilding {len(dimensions)} datasets...\n")

    for dim, (seeds, adv_target, ben_target) in dimensions.items():
        print(f"  → {dim.capitalize()}: {len(seeds)} seeds, target {adv_target} adv + {ben_target} benign")
        dataset = builder.build_dataset(
            dimension=dim,
            seeds=seeds,
            benign_controls=benign,
            expansion_factor=3,
            target_adversarial=adv_target,
            target_benign=ben_target
        )
        builder.save_dataset(dataset, f"data/final/{dim}.jsonl")
        
        # Count stats
        adv_count = sum(1 for p in dataset if p["prompt_type"] == "adversarial")
        ben_count = sum(1 for p in dataset if p["prompt_type"] == "benign")
        print(f"    Saved: {len(dataset)} total ({adv_count} adversarial, {ben_count} benign)")

    print("\nAll datasets built successfully!")
    print("   Files saved to data/final/")