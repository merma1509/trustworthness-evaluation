#!/usr/bin/env python3
"""Baseline validation: Runs sample seeds through LLM models and checks scoring

Tests 2-3 seeds per dimension against each model
Uses the same strict rubric as the main pipeline

Usage:
    python3 scripts/baseline_validation.py
    python3 scripts/baseline_validation.py --model gemma3:4b

Output:
    results/baseline_validation.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm_client import LLMClient
from src.utils import classify_response, classify_truthfulness


def load_seeds_from_raw() -> list:
    """Load 3 seeds per dimension from data/raw/*.jsonl for quick validation"""
    raw_dir = Path("data/raw")
    samples = []

    dimension_files = {
        "safety": "safety_seeds.jsonl",
        "truthfulness": "truthfulness_seeds.jsonl",
        "consistency": "consistency_seeds.jsonl",
        "benign": "benign_controls.jsonl",
    }

    samples_per_dim = {
        "safety": 3,
        "truthfulness": 3,
        "consistency": 2,
        "benign": 2,
    }

    for dimension, filename in dimension_files.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            print(f"  Skipping {filename} — file not found")
            continue
        with open(filepath) as f:
            seeds = [json.loads(line) for line in f]
        n = min(samples_per_dim.get(dimension, 2), len(seeds))
        for seed in seeds[:n]:
            seed["dimension"] = dimension
            samples.append(seed)
    return samples


def evaluate_model(model: str, seeds: list) -> dict:
    """Run all seeds against a single model using the strict rubric."""
    client = LLMClient(model=model, temperature=0.0, max_retries=2, timeout=30)

    if not client.check_health():
        return {"model": model, "status": "unavailable", "passed": 0, "failed": 0, "total": 0, "results": []}

    results = []
    passed = 0

    for seed in seeds:
        result = client.generate(seed["prompt_text"])
        dimension = seed.get("dimension", "unknown")
        attack_type = seed.get("attack_type", "")
        expected = seed["expected_behavior"]

        if not result["success"]:
            results.append({
                "prompt_id": seed["prompt_id"],
                "dimension": dimension,
                "expected": expected,
                "actual": "error",
                "passed": False,
                "scorer_label": "error",
                "scorer_explanation": f"API error: {result['error']}",
                "response": result["error"]
            })
            continue

        response_text = result["response"]  # Full response

        # Use the appropriate classifier based on dimension
        if dimension == "truthfulness":
            actual_label = classify_truthfulness(response_text, attack_type)
            # Map to is_correct
            if expected == "uncertain":
                is_correct = actual_label in ["correct", "not_attempted"]
            elif expected in ["comply", "safe_answer"]:
                is_correct = actual_label == "correct"
            else:
                is_correct = actual_label == expected
            
            explanation = f"classifier={actual_label}, expected={expected} -> {'correct' if is_correct else 'incorrect'}"
        else:
            # Safety and consistency use the standard classifier
            actual_label = classify_response(response_text)
            
            # Determine if correct
            is_correct = actual_label == expected
            if not is_correct and expected == "refuse" and actual_label in ["refuse", "uncertain"]:
                is_correct = True
            if not is_correct and expected in ["comply", "safe_answer"] and actual_label in ["comply"]:
                is_correct = True
            
            explanation = f"classifier={actual_label}, expected={expected} -> {'correct' if is_correct else 'incorrect'}"

        results.append({
            "prompt_id": seed["prompt_id"],
            "prompt_text": seed["prompt_text"],
            "dimension": dimension,
            "attack_type": attack_type,
            "expected": expected,
            "actual": actual_label,
            "passed": is_correct,
            "scorer_label": actual_label,
            "scorer_explanation": explanation,
            "response": response_text  # Full response, not truncated
        })

        if is_correct:
            passed += 1

    return {
        "model": model,
        "status": "completed",
        "total": len(seeds),
        "passed": passed,
        "failed": len(seeds) - passed,
        "pass_rate": f"{passed/len(seeds)*100:.0f}%" if len(seeds) > 0 else "0%",
        "results": results
    }


def print_results_table(all_results: list, seeds: list):
    """Print formatted comparison table."""
    print(f"\n{'='*120}")
    print("BASELINE VALIDATION RESULTS")
    print(f"{'='*120}")

    # Header
    header = f"{'Seed':<28} | {'Expected':<10} |"
    for r in all_results:
        header += f" {r['model'][:14]:<15} |"
    print(header)
    print("-" * 120)

    # Rows
    for i, seed in enumerate(seeds):
        row = f"{seed['prompt_id'][:28]:<28} | {seed['expected_behavior']:<10} |"
        for model_result in all_results:
            if model_result["status"] == "unavailable" or len(model_result["results"]) <= i:
                row += f" {'N/A':<15} |"
            else:
                sr = model_result["results"][i]
                symbol = "SUCCESS" if sr["passed"] else "FAIL"
                row += f" {symbol} {sr['actual']:<12} |"
        print(row)

    print("-" * 120)

    # Pass rate row
    rate_row = f"{'PASS RATE':<28} | {'':<10} |"
    for r in all_results:
        rate_row += f" {r['pass_rate']:<15} |"
    print(rate_row)
    print()


def print_summary(all_results: list):
    """Print per-model summary with scorer explanations."""
    print(f"{'='*120}")
    print("PER-MODEL SUMMARY")
    print(f"{'='*120}")

    for r in all_results:
        status = "COMPLETED" if r["status"] == "completed" else "FAILED"
        print(f"\n{status} {r['model']}")
        print(f"   Score: {r['passed']}/{r['total']} ({r['pass_rate']})")

        for res in r["results"]:
            status = "PASSED" if res["passed"] else "FAILED"
            print(f"   {status} {res['prompt_id']} ({res['dimension']})")
            print(f"      Expected: {res['expected']}, Got: {res['actual']}")
            print(f"      Scorer: {res['scorer_explanation']}")
            # Show first 100 chars of response for context
            preview = res['response'][:100].replace('\n', ' ')
            print(f"      Response: \"{preview}...\"")


def save_results(all_results: list, seeds: list):
    """Save to JSON with full response data."""
    output_path = Path("results/baseline_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "pipeline": "baseline_validation.py",
        "source": "data/raw/",
        "total_seeds": len(seeds),
        "results": all_results
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline validation of LLMs on trustworthiness seeds")
    parser.add_argument("--model", type=str, default="gemma3:4b",
                       help="Ollama model to test (default: gemma3:4b)")
    parser.add_argument("--all-models", action="store_true",
                       help="Test against all available models (gemma3:4b and llama3.1:8b)")
    args = parser.parse_args()

    print("=" * 60)
    print("TrustScore — Baseline Validation")
    print("=" * 60)

    # Load seeds
    seeds = load_seeds_from_raw()
    print(f"\nLoaded {len(seeds)} sample seeds from data/raw/")
    if len(seeds) == 0:
        print("No seeds found. Run 'python scripts/restructure_datasets.py' first.")
        return

    # Print what we're testing
    print("\nSeeds to test:")
    for s in seeds:
        dim = s.get("dimension", "").ljust(15)
        pid = s["prompt_id"].ljust(22)
        atype = s.get("attack_type", "").ljust(20)
        print(f"  {dim} {pid} {atype} → expected: {s['expected_behavior']}")

    # Determine models to test
    if args.all_models:
        models = ["gemma3:4b", "llama3.1:8b"]
    else:
        models = [args.model]

    # Evaluate each model
    all_results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")
        
        result = evaluate_model(model, seeds)
        all_results.append(result)

        if result["status"] == "unavailable":
            print(f"Model {model} is unavailable.")
            continue

        # Show quick results
        print(f"\nScore: {result['passed']}/{result['total']} ({result['pass_rate']})")

    # Print comparison if multiple models
    if len(all_results) > 1:
        print_results_table(all_results, seeds)
    
    print_summary(all_results)
    save_results(all_results, seeds)


if __name__ == "__main__":
    main()