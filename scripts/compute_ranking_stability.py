#!/usr/bin/env python3
"""compute_ranking_stability.py — ranking stability (flip probability)

Reproducibility driver ("Ranking stability (flip
probability)"). Reads the per-model dimension point scores (from
``results/scores_report.json``) and bootstraps, under each weight config, the
probability that model B outscores model A (`flip_probability`).

Single-source-of-truth: this is the ONE writer of
``results/ranking_stability.json``. It reuses ``src.stats.compute_ranking_stability``
(the same bootstrap engine used by ``run_evaluation.py``) so there is no
duplicated ranking logic. To stay fully compatible with the committed
consumers (``app/tabs/research_question.py`` dashboard tab, which reads the
``configurations`` block) it ALSO emits the ``configurations`` array in
exactly the shape ``run_evaluation.py`` produces: each entry has
``{"config", "ranking", "scores"}`` with colon-form model keys.

Determinism: bootstrap uses a fixed random seed (default 42), so a given
scores_report always yields the same result.

Usage:
    python3 scripts/compute_ranking_stability.py \
        --scores results/scores_report.json \
        --n-bootstrap 10000 \
        --output results/ranking_stability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import (  # noqa: E402
    DEFAULT_WEIGHT_CONFIGS,
    compute_ranking_stability,
    compute_weight_sensitivity,
)


def _dim_scores(per_model: Dict, model: str) -> Dict:
    """Extract the three dimension point scores for a model key."""
    s = per_model[model]
    return {
        "safety": s["safety"]["score"],
        "truthfulness": s["truthfulness"]["score"],
        "consistency": s["consistency"]["score"],
    }


def _colon_key(m: str) -> str:
    """Convert an underscore model label back to its colon-form Ollama key."""
    # results keys are like 'gemma3_4b' / 'llama3.1_8b' (from raw-output filenames);
    # the dashboard (research_question.py) expects 'gemma3:4b' colon-form keys.
    # Replace ONLY the first underscore so 'gemma3_4b' -> 'gemma3:4b'
    return m.replace("_", ":", 1)


def build_configurations(per_model: Dict, models: list) -> list:
    """Build the dashboard-compatible ``configurations`` array.

    Mirrors ``run_evaluation.compare_models`` exactly: for each weight config,
    compute each model's weighted score, rank descending, and store
    ``{"config", "ranking", "scores"}`` with colon-form model keys.
    """
    configurations = []
    for config in DEFAULT_WEIGHT_CONFIGS:
        model_scores = {}
        for model in models:
            dim = _dim_scores(per_model, model)
            model_scores[_colon_key(model)] = round(
                config["w_s"] * dim["safety"]
                + config["w_t"] * dim["truthfulness"]
                + config["w_c"] * dim["consistency"],
                4,
            )
        sorted_models = sorted(
            model_scores.items(), key=lambda x: x[1], reverse=True
        )
        ranking = {m: i + 1 for i, (m, _) in enumerate(sorted_models)}
        configurations.append(
            {"config": config["name"], "ranking": ranking, "scores": model_scores}
        )
    return configurations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True,
                        help="Path to results/scores_report.json.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/ranking_stability.json")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        print(f"  ✗ Scores report not found: {scores_path}. Run compute_scores.py first.")
        return 1

    scores = json.loads(scores_path.read_text())
    per_model = scores.get("per_model", {})
    if len(per_model) < 2:
        print("  ✗ Need at least 2 models in the scores report to compare rankings.")
        return 1

    # Pick the first two models, ordered alphabetically (stable / deterministic).
    models = sorted(per_model.keys())
    m1, m2 = models[0], models[1]

    # Bootstrap flip probability (from the SAME engine as run_evaluation.py).
    flip = compute_ranking_stability(
        _dim_scores(per_model, m1),
        _dim_scores(per_model, m2),
        weight_configs=DEFAULT_WEIGHT_CONFIGS,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )

    # Dashboard-compatible configurations (exact run_evaluation.py shape).
    configurations = build_configurations(per_model, models)

    # Single report: rich bootstrap output + backward-compatible view, plus a
    # convenience weight sensitivity summary for downstream verification.
    result = {
        "description": "Model ranking under different weight configurations",
        "configurations": configurations,
        "model1_key": m1,
        "model2_key": m2,
        "model1": _colon_key(m1),
        "model2": _colon_key(m2),
        "flip_probability": flip,
        "weight_sensitivity": {
            _colon_key(m): compute_weight_sensitivity(
                per_model[m]["safety"]["score"],
                per_model[m]["truthfulness"]["score"],
                per_model[m]["consistency"]["score"],
                DEFAULT_WEIGHT_CONFIGS,
            )
            for m in models
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Ranking stability written to {out_path}")
    print(f"    Comparing {_colon_key(m1)} vs {_colon_key(m2)}:")
    print(f"      overall model1 wins {flip['model_wins']['model1_pct']}%, "
          f"model2 wins {flip['model_wins']['model2_pct']}%")
    for cfg in flip["per_config"]:
        print(f"      {cfg['name']}: flip_prob={cfg['flip_probability']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
