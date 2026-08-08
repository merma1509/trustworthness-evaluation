#!/usr/bin/env python3
"""Main entry point for Trustworthness evaluation pipeline

Usage:
    python3 run_evaluation.py --models gemma3:4b,llama3.1:8b --output results/

Runs the full evaluation pipeline:
    1. Load datasets
    2. Evaluate all models across all 3 dimensions
    3. Compute per-model scores, CIs, weight sensitivity
    4. Paired model comparison (clustered bootstrap for consistency)
    5. Ranking stability across weight configurations
    6. Save all results
"""

import hashlib
import json
import subprocess
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.llm_client import LLMClient
from src.safety import evaluate_safety
from src.truthfulness import evaluate_truthfulness
from src.consistency import evaluate_consistency
from src.trustscore import compute_trustscore
from src.utils import save_jsonl
from src.stats import (
    DEFAULT_WEIGHT_CONFIGS,
    compute_paired_difference_ci,
    compute_clustered_consistency_ci,
)


# ──────────────────────────────────────────────────────────────
# Reproducibility helpers
# ──────────────────────────────────────────────────────────────


def _git_info() -> dict:
    """Retrieve git commit hash, branch, and tag at runtime.

    Returns:
        Dict with keys: commit_hash, branch, tag (or None).
    """
    info = {"commit_hash": None, "branch": None, "tag": None}
    try:
        info["commit_hash"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
            )
            .stdout.strip()
        )
        info["branch"] = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
            )
            .stdout.strip()
        )
        # Try to get the nearest tag
        tag = (
            subprocess.run(
                ["git", "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
            )
            .stdout.strip()
        )
        info["tag"] = tag if tag else None
    except Exception:
        pass
    return info


def _ollama_versions() -> list[dict]:
    """Retrieve local Ollama model versions (name, digest, size)."""
    models = []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 4:
                models.append({
                    "name": parts[0],
                    "id": parts[1],
                    "size": parts[2],
                    "modified": parts[3],
                })
    except Exception:
        pass
    return models


def _dependency_versions() -> dict[str, str]:
    """Return a dict of package -> version for key dependencies."""
    import importlib.metadata
    packages = [
        "sentence-transformers", "numpy", "scikit-learn",
        "streamlit", "ollama", "requests", "altair",
    ]
    versions = {}
    for pkg in packages:
        try:
            v = importlib.metadata.version(pkg)
            versions[pkg] = v
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _dataset_checksum(dataset_paths: list[str]) -> dict[str, str]:
    """Compute SHA-256 checksums for dataset files."""
    checksums = {}
    for path in dataset_paths:
        p = Path(path)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            checksums[p.name] = h
    return checksums


def create_manifest(
    models: list,
    output_dir: str,
    dataset_paths: list[str] | None = None,
    extra_args: dict | None = None,
):
    """Create a detailed manifest file for reproducibility.

    Includes: git commit, branch, tag, Ollama model digests,
    dependency versions, dataset checksums, CLI arguments.
    """
    manifest_path = Path(output_dir) / "manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    git = _git_info()
    ollama_models = _ollama_versions()
    deps = _dependency_versions()

    lines = [
        "=" * 72,
        "  Trustworthness Evaluation — Model Manifest",
        "=" * 72,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "── Git ──",
        f"  Commit:    {git['commit_hash'] or 'N/A'}",
        f"  Branch:    {git['branch'] or 'N/A'}",
        f"  Tag:       {git['tag'] or '(no tag)'}",
        "",
        "── Models Evaluated ──",
    ]
    for m in models:
        lines.append(f"    - {m}")
    lines.append("")

    if ollama_models:
        lines.append("── Local Ollama Models ──")
        lines.append(f"  {'Name':<30} {'Digest':<16} {'Size':<10}")
        lines.append(f"  {'-'*56}")
        for m in ollama_models:
            lines.append(f"  {m['name']:<30} {m['id']:<16} {m['size']:<10}")
        lines.append("")

    if deps:
        lines.append("── Python Dependencies ──")
        for pkg, ver in sorted(deps.items()):
            lines.append(f"  {pkg}=={ver}")
        lines.append("")

    checksums = {}
    if dataset_paths:
        checksums = _dataset_checksum(dataset_paths)
        if checksums:
            lines.append("── Dataset Checksums (SHA-256, first 16 hex) ──")
            for name, ck in checksums.items():
                lines.append(f"  {name}: {ck}")
            lines.append("")

    if extra_args:
        lines.append("── CLI Arguments ──")
        for k, v in extra_args.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.extend([
        "── Reproduction Command ──",
        f"  python3 run_evaluation.py \\",
        f"      --models {','.join(models)} \\",
        f"      --output {output_dir} \\",
        f"      --temperature {extra_args.get('temperature', 0.0)} \\",
        f"      --seed {extra_args.get('seed', 42)} \\",
        f"      --num-predict {extra_args.get('num_predict', 512)} \\",
        f"      --similarity-threshold {extra_args.get('similarity_threshold', 0.85)}",
    ])
    lines.append("")

    text = "\n".join(lines)

    with open(manifest_path, "w") as f:
        f.write(text + "\n")
    print(f"  Manifest saved to {manifest_path}")
    # Print summary to console
    print(f"    Git: {git['commit_hash'][:12] or '?'} @ {git['branch'] or '?'} "
          f"(tag: {git['tag'] or 'none'})")
    print(f"    Ollama: {len(ollama_models)} models available")
    print(f"    Dataset checksums: {list(checksums.values()) if dataset_paths else 'N/A'}")


def _load_raw(output_dir: str, model_label: str, dim: str) -> list | None:
    """Load raw-output JSONL records for a (model, dimension) pair."""
    path = Path(output_dir) / "raw_outputs" / f"{model_label}_{dim}.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _groups_from_records(records: list) -> dict[str, bool]:
    """Extract per-group consistency booleans from consistency raw output."""
    grps = defaultdict(list)
    for r in records:
        gid = r.get("group_id", r.get("prompt_id", "unknown"))
        grps[gid].append(r)
    result = {}
    for gid, recs in grps.items():
        if len(recs) == 1:
            continue  # exclude singletons
        result[gid] = any(r.get("group_consistent", False) for r in recs)
    return result


def run_paired_comparison(output_dir: str, model_names: list[str]) -> dict:
    """Compute paired / clustered bootstrap CIs for all dimensions."""
    model_labels = [m.replace(":", "_") for m in model_names]
    paired_results = {}

    for dim in ["safety", "truthfulness", "consistency"]:
        # Load both models
        recs = {}
        for label in model_labels:
            recs[label] = _load_raw(output_dir, label, dim)
        if any(v is None for v in recs.values()):
            print(f"  [SKIP] {dim}: raw outputs missing for one or both models")
            continue

        if dim == "consistency":
            m1_groups = _groups_from_records(recs[model_labels[0]])
            m2_groups = _groups_from_records(recs[model_labels[1]])
            ci_result = compute_clustered_consistency_ci(m1_groups, m2_groups)
        else:
            # Align by prompt_id — only compare prompts both models answered
            m1_by_id = {r.get("prompt_id", f"idx_{i}"): r
                        for i, r in enumerate(recs[model_labels[0]])}
            m2_by_id = {r.get("prompt_id", f"idx_{i}"): r
                        for i, r in enumerate(recs[model_labels[1]])}
            common_ids = sorted(set(m1_by_id.keys()) & set(m2_by_id.keys()))
            m1_scores = [1.0 if m1_by_id[pid].get("is_correct", False) else 0.0
                         for pid in common_ids]
            m2_scores = [1.0 if m2_by_id[pid].get("is_correct", False) else 0.0
                         for pid in common_ids]
            ci_result = compute_paired_difference_ci(m1_scores, m2_scores)

        paired_results[dim] = ci_result

    return paired_results


def compare_models(all_results: dict) -> dict:
    """Compare models and compute ranking stability across weight configs."""
    ranking_stability = {
        "description": "Model ranking under different weight configurations",
        "configurations": [],
    }

    models = list(all_results.keys())
    if len(models) < 2:
        return ranking_stability

    for config in DEFAULT_WEIGHT_CONFIGS:
        model_scores = {}
        for model in models:
            for ws_entry in all_results[model]["weight_sensitivity"]:
                if ws_entry["name"] == config["name"]:
                    model_scores[model] = ws_entry["score"]
                    break

        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = {m: i + 1 for i, (m, _) in enumerate(sorted_models)}

        ranking_stability["configurations"].append({
            "config": config["name"],
            "ranking": ranking,
            "scores": model_scores,
        })

    return ranking_stability


def print_ranking_warning(all_results: dict, models: list):
    """Print a warning about ranking instability."""
    if len(models) < 2:
        return

    print(f"\n  {'=' * 60}")
    print(f"  RANKING STABILITY — WARNING")
    print(f"  {'=' * 60}")

    names = []
    s_scores = []
    t_scores = []
    c_scores = []

    for model in models:
        if model in all_results:
            mr = all_results[model]
            names.append(model.replace(":", " "))
            s_scores.append(mr["dimension_scores"]["safety"]["score"])
            t_scores.append(mr["dimension_scores"]["truthfulness"]["score"])
            c_scores.append(mr["dimension_scores"]["consistency"]["score"])

    for i, name in enumerate(names):
        print(f"  {name}:")
        print(f"    Safety:       {s_scores[i]:.4f}")
        print(f"    Truthfulness: {t_scores[i]:.4f}")
        print(f"    Consistency:  {c_scores[i]:.4f}")
        print()

    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for dim_label, scores in [("Safety", s_scores),
                                          ("Truthfulness", t_scores),
                                          ("Consistency", c_scores)]:
                    if scores[i] > scores[j]:
                        print(f"  -> {names[i]} better on {dim_label}")
                    elif scores[j] > scores[i]:
                        print(f"  -> {names[j]} better on {dim_label}")
                    else:
                        print(f"  -> Tie on {dim_label}")

    print()
    print(f"  The overall TrustScore ranking depends heavily on weights.")
    print(f"  -> Do NOT claim a single 'winner'.")
    print(f"  -> Always report ranking as weight-dependent.")
    print(f"  -> See weight_sensitivity.json & paired_comparison.json.")
    print(f"  {'=' * 60}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Trustworthness Evaluation Pipeline"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma3:4b,llama3.1:8b",
        help="Comma-separated list of Ollama models to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (0.0 = deterministic)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Consistency cosine similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="final",
        help="Dataset version directory under data/ (default: 'final')",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="Max tokens per generation (default: 512)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K sampling (default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repeat penalty (default: 1.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    output_dir = args.output
    similarity_threshold = args.similarity_threshold
    dataset_version = args.dataset_version

    # Resolve dataset paths
    data_dir = Path("data") / dataset_version
    dataset_paths = {
        "safety": str(data_dir / "safety.jsonl"),
        "truthfulness": str(data_dir / "truthfulness.jsonl"),
        "consistency": str(data_dir / "consistency.jsonl"),
    }

    # Check that all dataset files exist
    for dim, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"  Dataset not found: {path}")
            print(f"  Available versions under data/:")
            for d in sorted(Path("data").iterdir()):
                if d.is_dir():
                    print(f"    - {d.name}/")
            sys.exit(1)

    print("=" * 60)
    print("Trustworthness Evaluation Pipeline")
    print(f"Models: {models}")
    print(f"Output: {output_dir}/")
    print(f"Dataset version: {dataset_version}")
    print(f"Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    git_info = _git_info()
    create_manifest(
        models,
        output_dir,
        dataset_paths=list(dataset_paths.values()),
        extra_args={
            "dataset_version": dataset_version,
            "similarity_threshold": similarity_threshold,
            "temperature": args.temperature,
        },
    )

    all_results = {}

    # ── Per-model evaluation ──────────────────────────────
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model}")
        print(f"{'=' * 60}")

        client = LLMClient(
            model=model,
            temperature=args.temperature,
            seed=args.seed,
            num_predict=args.num_predict,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            timeout=120,
        )
        if not client.check_health():
            print(f"  Model {model} unavailable. Skipping.")
            continue

        safety_result = evaluate_safety(
            client,
            dataset_path=dataset_paths["safety"],
            output_path=(
                f"{output_dir}/raw_outputs/"
                f"{model.replace(':', '_')}_safety.jsonl"
            ),
        )

        truthfulness_result = evaluate_truthfulness(
            client,
            dataset_path=dataset_paths["truthfulness"],
            output_path=(
                f"{output_dir}/raw_outputs/"
                f"{model.replace(':', '_')}_truthfulness.jsonl"
            ),
        )

        consistency_result = evaluate_consistency(
            client,
            dataset_path=dataset_paths["consistency"],
            output_path=(
                f"{output_dir}/raw_outputs/"
                f"{model.replace(':', '_')}_consistency.jsonl"
            ),
            similarity_threshold=similarity_threshold,
        )

        model_results = compute_trustscore(
            safety_result,
            truthfulness_result,
            consistency_result,
            weight_configs=DEFAULT_WEIGHT_CONFIGS,
            output_dir=f"{output_dir}/{model.replace(':', '_')}",
        )

        all_results[model] = model_results

        print(f"\n  {'=' * 40}")
        print(f"  {model} — RESULTS")
        print(f"  {'=' * 40}")
        print(f"  Safety:       "
              f"{model_results['dimension_scores']['safety']['score']:.4f}")
        print(f"  Truthfulness: "
              f"{model_results['dimension_scores']['truthfulness']['score']:.4f}")
        print(f"  Consistency:  "
              f"{model_results['dimension_scores']['consistency']['score']:.4f}")
        print(f"  ─────────────────────────────")
        print(f"  TrustScore:   "
              f"{model_results['trustworthiness_score']:.4f}")
        print(f"  {'=' * 40}")

    # ── Paired model comparison ───────────────────────────
    if len(all_results) >= 2:
        model_names = list(all_results.keys())
        paired_results = run_paired_comparison(output_dir, model_names)

        if paired_results:
            comparison_output = {
                "comparison": {
                    "model1": model_names[0],
                    "model2": model_names[1],
                    "method": (
                        "Paired bootstrap CI (clustered for consistency)"
                    ),
                },
                "dimensions": paired_results,
            }
            save_jsonl(
                [comparison_output],
                f"{output_dir}/paired_comparison.json",
            )

            print(f"\n  {'=' * 60}")
            print(f"  PAIRED MODEL COMPARISON")
            print(f"  {'=' * 60}")
            for dim, cr in paired_results.items():
                n = cr.get("n_pairs") or cr.get("n_groups", "?")
                print(
                    f"  {dim.upper()}: diff={cr['mean_difference']:+.4f}  "
                    f"[{cr['ci_lower']:.4f}, {cr['ci_upper']:.4f}]  "
                    f"p={cr['p_value']:.4f}  n={n}"
                )
            print(f"  -> Results saved to {output_dir}/paired_comparison.json")

    # ── Ranking stability ─────────────────────────────────
    if len(all_results) >= 2:
        ranking = compare_models(all_results)
        save_jsonl(
            [ranking],
            f"{output_dir}/ranking_stability.json",
        )
        print(f"\n  Ranking stability saved to "
              f"{output_dir}/ranking_stability.json")

    # ── Combined summary ──────────────────────────────────
    checksums = _dataset_checksum(list(dataset_paths.values()))
    combined = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "2.0",
        "git": _git_info(),
        "models": models,
        "dataset_version": dataset_version,
        "dataset_checksums": checksums,
        "temperature": args.temperature,
        "similarity_threshold": similarity_threshold,
        "results": all_results,
    }
    save_jsonl([combined], f"{output_dir}/results_summary.json")

    # ── Final warnings ────────────────────────────────────
    print_ranking_warning(all_results, models)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"All results saved to {output_dir}/")
    print(f"\nTo reproduce:")
    print(f"  python3 run_evaluation.py "
          f"--models {','.join(models)} --output {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

