#!/usr/bin/env python3
"""generate_results_json.py — assemble final result reports from score data.

Implements ``results/`` layout by deriving standalone result
reports from ``results/scores_report.json``:

  * ``trustscore_report.json`` — TrustScore + baseline weights per model.

Many of the other reports (agreement, CI, ranking, cost) are produced by their
own dedicated drivers; this script closes the small gaps (currently just the
TrustScore report) so every artefact has a stable home.

Usage:
    python3 scripts/generate_results_json.py \
        --scores results/scores_report.json \
        --output-results results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True,
                        help="Path to results/scores_report.json.")
    parser.add_argument("--output-results", default="results",
                        help="Directory to write derived reports (default: results).")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        print(f"  ✗ scores report not found: {scores_path}. Run compute_scores.py first.")
        return 1

    scores = json.loads(scores_path.read_text())
    out_dir = Path(args.output_results)

    # TrustScore report per model.
    trust_path = out_dir / "trustscore_report.json"
    trust_report = {
        "pipeline": "scripts/generate_results_json.py",
        "source": str(scores_path),
        "trustscore_by_model": scores.get("trustscore_by_model", {}),
        "weights_default": scores.get("weights_default", []),
    }
    with trust_path.open("w") as f:
        json.dump(trust_report, f, indent=2, ensure_ascii=False)
    print(f"  TrustScore report written to {trust_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
