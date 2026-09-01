#!/usr/bin/env python3
"""compute_ci.py — bootstrap confidence intervals from per-prompt raw scores.

Reproducibility driver("Bootstrap CIs (n=10000)").
Reads the per-model/per-dimension rescored results *again* from raw outputs
(via the same rescore functions) so the CI bootstrap and the point scores are
computed from the same immutable inputs, then writes ``results/ci_report.json``.

Determinism: bootstrap uses a fixed RNG seed (default 42), so a given input set
always yields the same CI

Usage:
    python3 scripts/compute_ci.py \
        --raw results/raw_outputs \
        --n-bootstrap 10000 \
        --output results/ci_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import compute_confidence_intervals  # noqa: E402


def _load_rescore_module():
    spec = importlib.util.spec_from_file_location(
        "_score_saved_outputs", Path(__file__).parent / "score_saved_outputs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _per_prompt_trials(dim: str, result: dict) -> List[float]:
    """Extract 0/1 per-independent-unit trials from a rescored dimension result.

    Mirrors ``compute_trust_score`` in score_saved_outputs.py: safety/truthfulness
    use per-prompt correctness (excl. benign truthfulness), consistency uses one
    trial per non-singleton group.
    """
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


def compute_ci(raw_dir: Path, output_path: Path, n_bootstrap: int) -> Dict:
    rc = _load_rescore_module()
    suffix_dim = {"safety": "safety", "truthfulness": "truthfulness", "consistency": "consistency"}

    # Re-rescore each model×dimension to get per-prompt results for the CI.
    files = sorted(raw_dir.glob("*_safety.jsonl")) + sorted(
        raw_dir.glob("*_truthfulness.jsonl")
    ) + sorted(raw_dir.glob("*_consistency.jsonl"))

    by_model: Dict[str, Dict[str, dict]] = {}
    for f in files:
        stem = f.stem
        dim = next((d for s, d in suffix_dim.items() if stem.endswith("_" + s)), None)
        if dim is None:
            continue
        model = stem.rsplit("_", 1)[0]
        records = _load_jsonl(f)
        if dim == "safety":
            res = rc.rescore_safety(records)
        elif dim == "truthfulness":
            res = rc.rescore_truthfulness(records)
        else:
            res = rc.rescore_consistency(records)
        by_model.setdefault(model, {})[dim] = res

    report: Dict = {
        "pipeline": "scripts/compute_ci.py",
        "n_bootstrap": n_bootstrap,
        "ci_level": 0.95,
        "seed": 42,
        "per_model": {},
    }
    for model, dims in sorted(by_model.items()):
        report["per_model"][model] = {}
        for dim in ("safety", "truthfulness", "consistency"):
            if dim not in dims:
                continue
            trials = _per_prompt_trials(dim, dims[dim])
            if not trials:
                report["per_model"][model][dim] = {
                    "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0
                }
                continue
            ci = compute_confidence_intervals(trials, n_bootstrap=n_bootstrap)
            ci.pop("note", None)
            report["per_model"][model][dim] = ci

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  CI report written to {output_path}")
    for model, dims in report["per_model"].items():
        for dim, ci in dims.items():
            print(f"    {model}/{dim}: {ci['mean']} [{ci['ci_lower']}, {ci['ci_upper']}] n={ci['n']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="results/raw_outputs")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--output", default="results/ci_report.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        print(f"  Raw outputs dir not found: {raw_dir}")
        return 1
    compute_ci(raw_dir, Path(args.output), args.n_bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
