#!/usr/bin/env python3
"""compute_paired_comparison.py — paired/clustered model comparison (RQ2).

Significance testing between the two models must respect the *design*:

* **safety / truthfulness**: each prompt yields one 0/1 score per model, and the
  two models answer the *same* prompts → the correct test is a **paired
  bootstrap** (``compute_paired_difference_ci``), not overlap of marginal CIs.
* **consistency**: evaluated at the **group** level (each group is a cluster of
  semantically related prompts) → the correct test is a **clustered bootstrap**
  (``compute_clustered_consistency_ci``) that resamples whole groups.

The paired/clustered results are written to ``results/paired_comparison.json``
(the file the dashboard consumes for its "Significance of Model Differences"
block).

Determinism: resampling is seeded (default 42), so the output is reproducible
from the raw outputs.

Usage:
    python3 scripts/compute_paired_comparison.py \
        --raw results/raw_outputs \
        --output results/paired_comparison.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action  # noqa: E402
from src.stats import (  # noqa: E402
    compute_clustered_consistency_ci,
    compute_paired_difference_ci,
)

DIMENSIONS = ("safety", "truthfulness", "consistency")


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


def _per_prompt_trials(dim: str, result: dict) -> List[float]:
    """Per-independent-unit 0/1 trials (mirrors compute_ci.py exactly)."""
    trials: List[float] = []
    seen_groups = set()
    for r in result.get("results", []):
        if dim == "consistency":
            gid = r.get("group_id")
            if not gid or gid in seen_groups:
                continue
            seen_groups.add(gid)
            if r.get("is_singleton", False):
                continue
            trials.append(1.0 if r.get("group_consistent") else 0.0)
        else:
            if r.get("is_benign", False):
                continue
            trials.append(1.0 if r.get("is_correct") else 0.0)
    return trials


def _consistency_group_map(result: dict) -> Dict[str, bool]:
    """Map group_id -> bool(consistent) for the clustered bootstrap."""
    out: Dict[str, bool] = {}
    for r in result.get("results", []):
        gid = r.get("group_id")
        if not gid or r.get("is_singleton", False):
            continue
        out[gid] = bool(r.get("group_consistent"))
    return out


def compute_paired_comparison(raw_dir: Path, output_path: Path, seed: int = 42) -> Dict:
    """Compare the two models per dimension with the design-correct test."""
    rc = _load_rescore_module()
    files = sorted(raw_dir.glob("*_safety.jsonl")) + sorted(
        raw_dir.glob("*_truthfulness.jsonl")
    ) + sorted(raw_dir.glob("*_consistency.jsonl"))

    models: Dict[str, Dict[str, dict]] = {}
    for f in files:
        stem = f.stem
        dim = None
        for suffix, dname in {"safety": "safety", "truthfulness": "truthfulness",
                              "consistency": "consistency"}.items():
            if stem.endswith("_" + suffix):
                dim = dname
                break
        if dim is None:
            continue
        model = stem.rsplit("_", 1)[0]
        records = _load_jsonl(f)
        if dim == "safety":
            models.setdefault(model, {})[dim] = rc.rescore_safety(records)
        elif dim == "truthfulness":
            models.setdefault(model, {})[dim] = rc.rescore_truthfulness(records)
        else:
            models.setdefault(model, {})[dim] = rc.rescore_consistency(records)

    if len(models) < 2:
        raise SystemExit("Need at least 2 models in raw outputs to compare.")
    m1, m2 = sorted(models.keys())[0], sorted(models.keys())[1]

    dimensions: Dict[str, dict] = {}
    for dim in DIMENSIONS:
        r1, r2 = models[m1].get(dim), models[m2].get(dim)
        if dim == "consistency":
            g1 = _consistency_group_map(r1) if r1 else {}
            g2 = _consistency_group_map(r2) if r2 else {}
            res = compute_clustered_consistency_ci(g1, g2, n_bootstrap=10000, ci=0.95)
            dimensions[dim] = {
                "mean_difference": res["mean_difference"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "p_value": res["p_value"],
                "n_groups": res["n_groups"],
                "method": "clustered bootstrap (resample whole groups)",
            }
        else:
            t1 = _per_prompt_trials(dim, r1) if r1 else []
            t2 = _per_prompt_trials(dim, r2) if r2 else []
            res = compute_paired_difference_ci(t1, t2, n_bootstrap=10000, ci=0.95)
            dimensions[dim] = {
                "mean_difference": res["mean_difference"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "p_value": res["p_value"],
                "n_pairs": res["n_pairs"],
                "method": "paired bootstrap (same prompts per model)",
            }

    report = {
        "pipeline": "scripts/compute_paired_comparison.py",
        "source": str(raw_dir),
        "seed": seed,
        "n_bootstrap": 10000,
        "comparison": {
            "model1": m1.replace("_", ":"),
            "model2": m2.replace("_", ":"),
            "method": "Paired bootstrap CI (clustered for consistency)",
        },
        "dimensions": dimensions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False)

    print(f"  Paired/clustered comparison written to {output_path}")
    for dim, d in dimensions.items():
        print(
            f"    {dim}: diff={d['mean_difference']}, CI=[{d.get('ci_lower')}, "
            f"{d.get('ci_upper')}], p={d['p_value']} "
            f"(n={d.get('n_pairs') or d.get('n_groups')})"
        )

    log_action(
        log_path=Path("experiment/logs/processing_log.jsonl"),
        action="compute_paired_comparison",
        script="scripts/compute_paired_comparison.py",
        args={"--raw": str(raw_dir), "--seed": seed, "--output": str(output_path)},
        input_paths={"raw": raw_dir},
        output_paths={"output": output_path},
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="results/raw_outputs",
                        help="Directory of {model}_{dim}.jsonl raw outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/paired_comparison.json")
    args = parser.parse_args()
    compute_paired_comparison(Path(args.raw), Path(args.output), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
