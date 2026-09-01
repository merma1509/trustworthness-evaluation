#!/usr/bin/env python3
"""compute_scores.py — compute per-model, per-dimension scores from raw outputs.

Composes the existing rescoring logic (``score_saved_outputs.py``) with the
offline reproducibility goal: given only the immutable
raw model outputs, reproduce each model's dimension scores and its combined
TrustScore.

Determinism: rescoring is a pure function of the raw outputs (no randomness),
so this script is byte-for-byte reproducible for the same inputs.

Usage:
    python3 scripts/compute_scores.py \
        --raw results/raw_outputs \
        --output results/scores_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import DEFAULT_WEIGHT_CONFIGS  # noqa: E402


def _load_rescore_module():
    """Import scripts/score_saved_outputs.py as a module without clobbering argv."""
    spec = importlib.util.spec_from_file_location(
        "_score_saved_outputs", Path(__file__).parent / "score_saved_outputs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def compute_scores(raw_dir: Path, output_path: Path) -> Dict:
    """Rescore every model×dimension raw file and assemble a scores report.

    Args:
        raw_dir: Directory containing ``{model}_{dim}.jsonl`` files (as produced
            by ``run_evaluation.py``).
        output_path: Where to write the per-model/per-dimension scores JSON.

    Returns:
        Dict with keys 'models', 'per_model', 'trustscore_by_model', and
        'weights' used for the basline aggregation.
    """
    rc = _load_rescore_module()

    # Discover raw output files: {model}_{dimension}.jsonl
    files = sorted(raw_dir.glob("*_safety.jsonl")) + sorted(
        raw_dir.glob("*_truthfulness.jsonl")
    ) + sorted(raw_dir.glob("*_consistency.jsonl"))

    # Group by model stem (strip the _dimension suffix).
    dimension_map = {  # suffix -> dimension name
        "safety": "safety",
        "truthfulness": "truthfulness",
        "consistency": "consistency",
    }
    models: Dict[str, Dict[str, dict]] = {}
    for f in files:
        stem = f.stem
        # Identify which dimension this file is.
        dim = None
        for suffix, dname in dimension_map.items():
            if stem.endswith("_" + suffix):
                dim = dname
                break
        if dim is None:
            continue
        model = stem.rsplit("_", 1)[0]
        records = _load_jsonl(f)
        if dim == "safety":
            result = rc.rescore_safety(records)
        elif dim == "truthfulness":
            result = rc.rescore_truthfulness(records)
        else:
            result = rc.rescore_consistency(records)
        models.setdefault(model, {})[dim] = result

    # Per-model trustscore + dimension scores summary.
    per_model: Dict[str, dict] = {}
    trustscore_by_model: Dict[str, dict] = {}
    for model, dims in sorted(models.items()):
        if not all(d in dims for d in ("safety", "truthfulness", "consistency")):
            continue
        per_model[model] = {
            dim: {
                "score": dims[dim]["score"],
                "correct": dims[dim].get("correct", 0),
                "total": dims[dim].get("total", dims[dim].get("total_groups", 0)),
            }
            for dim in ("safety", "truthfulness", "consistency")
        }
        trust = rc.compute_trust_score(
            dims["safety"], dims["truthfulness"], dims["consistency"]
        )
        per_model[model]["trust_score"] = trust["trustworthiness_score"]
        trustscore_by_model[model] = trust

    report = {
        "pipeline": "scripts/compute_scores.py",
        "source": str(raw_dir),
        "models": sorted(models.keys()),
        "per_model": per_model,
        "weights_default": DEFAULT_WEIGHT_CONFIGS,
        "trustscore_by_model": {
            m: {
                "trustworthiness_score": t["trustworthiness_score"],
                "baseline_weights": t["baseline_weights"],
                "dimension_scores": {
                    d: t["dimension_scores"][d]["score"]
                    for d in ("safety", "truthfulness", "consistency")
                },
            }
            for m, t in trustscore_by_model.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Scores written to {output_path}")
    for model, dims in per_model.items():
        print(
            f"    {model}: "
            + ", ".join(
                f"{d}={dims[d]['score']}" for d in ("safety", "truthfulness", "consistency")
            )
            + f" | trust={dims['trust_score']}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="results/raw_outputs",
                        help="Directory of {model}_{dim}.jsonl raw outputs.")
    parser.add_argument("--output", default="results/scores_report.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        print(f"  Raw outputs dir not found: {raw_dir}")
        return 1
    compute_scores(raw_dir, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
