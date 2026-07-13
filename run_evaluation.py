#!/usr/bin/env python3
"""Main entry point for Trustworthness evaluation pipeline
Usage:
    python3 run_evaluation.py --models gemma3:4b,llama3.1:8b --output results/

Runs the full evaluation pipeline:
    1. Load datasets
    2. Evaluate all models across all 3 dimensions
    3. Compute scores, confidence intervals, weight sensitivity
    4. Save all results
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.llm_client import LLMClient
from src.safety import evaluate_safety
from src.truthfulness import evaluate_truthfulness
from src.consistency import evaluate_consistency
from src.trustscore import compute_trustscore
from src.utils import save_jsonl, DEFAULT_WEIGHT_CONFIGS


def create_manifest(models: list, output_dir: str):
    """Create a model manifest file with exact versions and settings."""
    manifest_path = Path(output_dir) / "manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"Trustworthness Model Manifest",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Models:",
    ]
    for model in models:
        lines.append(f"  - {model}")
    lines.extend([
        f"",
        f"Inference Settings:",
        f"  temperature: 0.0 (deterministic)",
        f"  max_retries: 3",
        f"  timeout: 60s",
        f"",
        f"Reproduction Command:",
        f"  python3 run_evaluation.py --models {','.join(models)} --output {output_dir}",
    ])
    
    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Manifest saved to {manifest_path}")


def compare_models(all_results: dict) -> dict:
    """Compare models and compute ranking stability."""
    ranking_stability = {
        "description": "Spearman rank correlation between weight configurations",
        "note": "With only 2 models, ranking stability = 1.0 if rankings are consistent, 0.0 if they flip.",
        "configurations": []
    }
    
    models = list(all_results.keys())
    if len(models) < 2:
        return ranking_stability
    
    # For each weight config, check if ranking is the same
    for config in DEFAULT_WEIGHT_CONFIGS:
        model_scores = {}
        for model in models:
            for ws_entry in all_results[model]["weight_sensitivity"]:
                if ws_entry["name"] == config["name"]:
                    model_scores[model] = ws_entry["score"]
                    break
        
        # Rank models (1 = higher score)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = {m: i+1 for i, (m, _) in enumerate(sorted_models)}
        
        ranking_stability["configurations"].append({
            "config": config["name"],
            "ranking": ranking,
            "scores": model_scores
        })
    
    return ranking_stability


def print_ranking_warning(model_results: dict, all_results: dict, models: list):
    """Print a warning about ranking instability."""
    if len(models) < 2:
        return
    
    print(f"\n  {'='*50}")
    print(f"  RANKING INSTABILITY WARNING")
    print(f"  {'='*50}")
    
    # Get dimension scores
    model_names = []
    safety_scores = []
    truthfulness_scores = []
    consistency_scores = []
    
    for model in models:
        if model in all_results:
            mr = all_results[model]
            model_names.append(model.replace(':', ' '))
            safety_scores.append(mr['dimension_scores']['safety']['score'])
            truthfulness_scores.append(mr['dimension_scores']['truthfulness']['score'])
            consistency_scores.append(mr['dimension_scores']['consistency']['score'])
    
    # Print per-dimension comparison
    for i, name in enumerate(model_names):
        print(f"  {name}:")
        print(f"    Safety:       {safety_scores[i]:.4f}")
        print(f"    Truthfulness: {truthfulness_scores[i]:.4f}")
        print(f"    Consistency:  {consistency_scores[i]:.4f}")
        print()
    
    # Determine who wins each dimension
    if safety_scores[0] > safety_scores[1]:
        print(f"  -> {model_names[0]} appears better on Safety.")
    elif safety_scores[1] > safety_scores[0]:
        print(f"  -> {model_names[1]} appears better on Safety.")
    else:
        print(f"  -> Both models tie on Safety.")
    
    if truthfulness_scores[0] > truthfulness_scores[1]:
        print(f"  -> {model_names[0]} appears better on Truthfulness.")
    elif truthfulness_scores[1] > truthfulness_scores[0]:
        print(f"  -> {model_names[1]} appears better on Truthfulness.")
    else:
        print(f"  -> Both models tie on Truthfulness.")
    
    if consistency_scores[0] > consistency_scores[1]:
        print(f"  -> {model_names[0]} appears better on Consistency.")
    elif consistency_scores[1] > consistency_scores[0]:
        print(f"  -> {model_names[1]} appears better on Consistency.")
    else:
        print(f"  -> Both models tie on Consistency.")
    
    print()
    print(f"  The overall TrustScore ranking depends heavily on")
    print(f"  the chosen weight configuration.")
    print(f"  -> Do NOT claim a single 'winner'.")
    print(f"  -> Always report ranking as weight-dependent.")
    print(f"  -> See weight_sensitivity.json for the full analysis.")
    print(f"  {'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Trustworthness Evaluation Pipeline")
    parser.add_argument(
        "--models",
        type=str,
        default="gemma3:4b,llama3.1:8b",
        help="Comma-separated list of Ollama models to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (0.0 = deterministic)"
    )
    
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    output_dir = args.output
    
    print("=" * 60)
    print("Trustworthness Evaluation Pipeline")
    print(f"Models: {models}")
    print(f"Output: {output_dir}/")
    print(f"Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    create_manifest(models, output_dir)
    
    # Store all results
    all_results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model}")
        print(f"{'='*60}")
        
        # Initialize client with deterministic inference
        client = LLMClient(model=model, temperature=args.temperature)
        
        if not client.check_health():
            print(f"  Model {model} unavailable. Skipping.")
            continue
        
        # Run evaluation modules
        safety_result = evaluate_safety(
            client,
            dataset_path="data/final/safety.jsonl",
            output_path=f"{output_dir}/raw_outputs/{model.replace(':', '_')}_safety.jsonl"
        )
        
        truthfulness_result = evaluate_truthfulness(
            client,
            dataset_path="data/final/truthfulness.jsonl",
            output_path=f"{output_dir}/raw_outputs/{model.replace(':', '_')}_truthfulness.jsonl"
        )
        
        consistency_result = evaluate_consistency(
            client,
            dataset_path="data/final/consistency.jsonl",
            output_path=f"{output_dir}/raw_outputs/{model.replace(':', '_')}_consistency.jsonl"
        )
        
        # Compute trust score
        model_results = compute_trustscore(
            safety_result,
            truthfulness_result,
            consistency_result,
            weight_configs=DEFAULT_WEIGHT_CONFIGS,
            output_dir=f"{output_dir}/{model.replace(':', '_')}"
        )
        
        all_results[model] = model_results
        
        # Print summary
        print(f"\n  {'='*40}")
        print(f"  {model} — RESULTS")
        print(f"  {'='*40}")
        print(f"  Safety:       {model_results['dimension_scores']['safety']['score']:.4f}")
        print(f"  Truthfulness: {model_results['dimension_scores']['truthfulness']['score']:.4f}")
        print(f"  Consistency:  {model_results['dimension_scores']['consistency']['score']:.4f}")
        print(f"  ─────────────────────────────")
        print(f"  TrustScore:   {model_results['trustworthiness_score']:.4f}")
        print(f"  {'='*40}")
    
    # Compute ranking stability across models
    if len(all_results) >= 2:
        ranking = compare_models(all_results)
        save_jsonl([ranking], f"{output_dir}/ranking_stability.json")
        print(f"\n  Ranking stability saved to {output_dir}/ranking_stability.json")
    
    # Save combined results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "temperature": args.temperature,
        "results": all_results
    }
    save_jsonl([combined], f"{output_dir}/results_summary.json")
    
    # Print ranking instability warning
    print_ranking_warning(model_results, all_results, models)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to {output_dir}/")
    print(f"\nTo reproduce:")
    print(f"  python3 run_evaluation.py --models {','.join(models)} --output {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()