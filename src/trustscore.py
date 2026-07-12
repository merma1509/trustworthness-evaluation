"""TrustScore aggregator
Combines dimension scores into final trustworthiness scores
Computes: weighted sum, confidence intervals, weight sensitivity, ranking stability
Accepts confusion matrix from safety evaluation"""

from typing import List, Dict, Tuple
import json
from pathlib import Path
from src.utils import (
    compute_confidence_intervals,
    compute_weight_sensitivity,
    DEFAULT_WEIGHT_CONFIGS,
    save_jsonl
)


def compute_trustscore(
    safety_result: Dict,
    truthfulness_result: Dict,
    consistency_result: Dict,
    weight_configs: List[Dict] = None,
    output_dir: str = "results"
) -> Dict:
    """Compute final TrustScore from dimension evaluation results
    
    Args:
        safety_result: Output from evaluate_safety() with confusion_matrix
        truthfulness_result: Output from evaluate_truthfulness()
        consistency_result: Output from evaluate_consistency()
        weight_configs: List of weight configurations for sensitivity analysis
        output_dir: Directory to save results
        
    Returns:
        Dict with complete evaluation results
    """
    if weight_configs is None:
        weight_configs = DEFAULT_WEIGHT_CONFIGS
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract dimension scores
    s = safety_result["score"]
    t = truthfulness_result["score"]
    c = consistency_result["score"]
    
    # --- Dimension Scores ---
    dimension_scores = {
        "safety": {
            "score": s,
            "correct": safety_result["correct"],
            "total": safety_result["total"],
            "confusion_matrix": safety_result.get("confusion_matrix", {})
        },
        "truthfulness": {
            "score": t,
            "correct": truthfulness_result["correct"],
            "total": truthfulness_result["total"]
        },
        "consistency": {
            "score": c,
            "consistent_groups": consistency_result["consistent_groups"],
            "total_groups": consistency_result["total_groups"]
        },
    }
    
    # --- Confidence Intervals ---
    safety_trials = [1 if r["is_correct"] else 0 for r in safety_result["results"]]
    truthfulness_trials = [1 if r["is_correct"] else 0 for r in truthfulness_result["results"]]
    
    # Deduplicate consistency trials (one per group)
    seen_groups = set()
    consistency_group_trials = []
    for r in consistency_result["results"]:
        gid = r.get("group_id")
        if gid and gid not in seen_groups and "group_consistent" in r:
            seen_groups.add(gid)
            consistency_group_trials.append(1 if r["group_consistent"] else 0)
    
    confidence_intervals = {
        "safety": compute_confidence_intervals(safety_trials),
        "truthfulness": compute_confidence_intervals(truthfulness_trials),
        "consistency": compute_confidence_intervals(
            consistency_group_trials if consistency_group_trials else [c]
        ),
    }
    
    # --- Weight Sensitivity ---
    weight_sensitivity = compute_weight_sensitivity(s, t, c, weight_configs)
    
    # --- Baseline Score ---
    baseline = weight_configs[0]  # First config is baseline
    trustworthiness = round(
        baseline["w_s"] * s +
        baseline["w_t"] * t +
        baseline["w_c"] * c,
        4
    )
    
    # --- Assemble Results ---
    results = {
        "trustworthiness_score": trustworthiness,
        "baseline_weights": {
            "safety": baseline["w_s"],
            "truthfulness": baseline["w_t"],
            "consistency": baseline["w_c"]
        },
        "dimension_scores": dimension_scores,
        "confidence_intervals": confidence_intervals,
        "weight_sensitivity": weight_sensitivity,
        "weight_configs_tested": len(weight_configs)
    }
    
    # Save scores
    save_jsonl([results], str(output_path / "scores.json"))
    
    # Save confidence intervals separately
    save_jsonl([confidence_intervals], str(output_path / "confidence_intervals.json"))
    
    # Save weight sensitivity separately
    save_jsonl(weight_sensitivity, str(output_path / "weight_sensitivity.json"))
    
    return results