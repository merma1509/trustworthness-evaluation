"""validation.py
Measurement validation for the trustworthiness evaluation pipeline.
Implements the new paradigm: "When can we trust a small local evaluation?"

Provides:
    - compute_error_analysis():  per-factor breakdown of auto-scorer errors
    - compute_agreement_by_factor(): agreement stratified by prompt/model attributes
    - compute_measurement_budget(): human vs. auto cost comparison
"""

from collections import defaultdict
from typing import Dict, List, Optional

from src.agreement import compute_agreement


# ──────────────────────────────────────────────────────────────
# Error analysis: where does the auto-scorer disagree with human?
# ──────────────────────────────────────────────────────────────


def compute_error_analysis(
    records: List[dict],
) -> Dict:
    """Analyze auto-scorer errors by dimension, attack type, and model.

    Args:
        records: List of audit record dicts with human_label and auto_label.

    Returns:
        Dict with keys:
            'overall': agreement stats
            'by_dimension': {dim: agreement stats}
            'by_attack_type': {attack_type: agreement stats}
            'by_model': {model: agreement stats}
            'false_positives': examples where auto said correct, human disagreed
            'false_negatives': examples where auto said incorrect, human disagreed
    """
    labelled = [r for r in records if r.get("human_label") is not None]
    if not labelled:
        return {"error": "No human labels found"}

    result = {
        "overall": _agreement_from_list(labelled),
        "by_dimension": _group_agreement(labelled, "dimension"),
        "by_attack_type": _group_agreement(labelled, "attack_type"),
        "by_model": _group_agreement(labelled, "model"),
        "false_positives": [],
        "false_negatives": [],
    }

    # Collect false positives (auto better than human) and false negatives
    for r in labelled:
        h = r.get("human_label", "")
        a = r.get("auto_label", "")
        if h == a:
            continue

        if a in ("correct", "consistent") and h in ("incorrect", "inconsistent"):
            result["false_positives"].append(r)
        else:
            result["false_negatives"].append(r)

    return result


def _agreement_from_list(records: List[dict]) -> Dict:
    """Compute agreement from a list of records with human_label/auto_label."""
    pairs = [
        (r["human_label"], r["auto_label"])
        for r in records
        if r.get("human_label") is not None
    ]
    humans = [p[0] for p in pairs]
    autos = [p[1] for p in pairs]
    return compute_agreement(humans, autos)


def _group_agreement(records: List[dict], group_key: str) -> Dict:
    """Compute agreement stratified by a record field."""
    groups = defaultdict(list)
    for r in records:
        key = r.get(group_key, "unknown")
        groups[key].append(r)

    result = {}
    for key in sorted(groups):
        ag = _agreement_from_list(groups[key])
        result[key] = {
            "n": ag["n_valid_pairs"],
            "agreement_rate": ag["agreement_rate"],
            "cohens_kappa": ag["cohens_kappa"],
        }
    return result


def _get_label_vals(r: dict) -> tuple:
    """Extract numeric values for false positive / negative analysis."""
    import numpy as np
    h = r.get("human_label", "")
    a = r.get("auto_label", "")
    # Map to 1 (good) / 0 (bad)
    h_val = 1 if h in ("correct", "consistent") else 0
    a_val = 1 if a in ("correct", "consistent") else 0
    return h_val, a_val


# ──────────────────────────────────────────────────────────────
# Measurement cost: human vs. auto comparison
# ──────────────────────────────────────────────────────────────


def compute_measurement_budget(
    auto_time_per_prompt_seconds: float = 10.0,
    human_time_per_label_seconds: float = 30.0,
    num_prompts: int = 105,
    num_models: int = 2,
    cost_per_hour_human: float = 20.0,   # hourly wage for annotator
    cost_per_hour_gpu: float = 0.50,     # estimated GPU cost (local = ~0)
) -> Dict:
    """Compare cost of fully automatic vs. fully human evaluation.

    Args:
        auto_time_per_prompt_seconds: Average inference time per prompt.
        human_time_per_label_seconds: Average human annotation time per label.
        num_prompts: Number of prompts in dataset.
        num_models: Number of models being evaluated.
        cost_per_hour_human: Hourly cost for human annotator.
        cost_per_hour_gpu: Hourly GPU cost (local Ollama = negligible).

    Returns:
        Dict comparing auto vs. human on time, cost, and scalability.
    """
    total_responses = num_prompts * num_models

    # Auto
    auto_seconds = total_responses * auto_time_per_prompt_seconds
    auto_hours = auto_seconds / 3600
    auto_cost = auto_hours * cost_per_hour_gpu

    # Human
    human_seconds = total_responses * human_time_per_label_seconds
    human_hours = human_seconds / 3600
    human_cost = human_hours * cost_per_hour_human

    # Hybrid (50% auto + 50% human audit)
    audit_pct = 0.5
    audit_responses = int(total_responses * audit_pct)
    hybrid_auto_seconds = total_responses * auto_time_per_prompt_seconds
    hybrid_human_seconds = audit_responses * human_time_per_label_seconds
    hybrid_total_hours = (hybrid_auto_seconds + hybrid_human_seconds) / 3600
    hybrid_cost = (
        hybrid_auto_seconds / 3600 * cost_per_hour_gpu
        + hybrid_human_seconds / 3600 * cost_per_hour_human
    )

    return {
        "parameters": {
            "num_prompts": num_prompts,
            "num_models": num_models,
            "total_responses": total_responses,
            "auto_time_per_prompt_sec": auto_time_per_prompt_seconds,
            "human_time_per_label_sec": human_time_per_label_seconds,
            "human_hourly_cost": cost_per_hour_human,
            "gpu_hourly_cost": cost_per_hour_gpu,
        },
        "fully_automatic": {
            "time_hours": round(auto_hours, 2),
            "cost": round(auto_cost, 2),
            "n_responses": total_responses,
            "scalability_note": "O(n*m) — linear in models × prompts",
        },
        "fully_human": {
            "time_hours": round(human_hours, 2),
            "cost": round(human_cost, 2),
            "n_responses": total_responses,
            "scalability_note": "O(n*m) — same linear cost, but 60× more expensive",
        },
        "hybrid_50pct_audit": {
            "time_hours": round(hybrid_total_hours, 2),
            "cost": round(hybrid_cost, 2),
            "audit_responses": audit_responses,
            "auto_responses": total_responses,
            "note": "Run auto on all, human on 50% for validation",
        },
        "recommendation": (
            f"For {total_responses} responses across {num_models} models × {num_prompts} prompts, "
            f"automatic evaluation is **{human_cost / auto_cost:.0f}× cheaper** "
            f"than full human evaluation. "
            f"A hybrid approach (auto + 50% human audit) costs "
            f"${hybrid_cost:.0f} and provides measurement validation."
        ),
    }


# ──────────────────────────────────────────────────────────────
# Full validation report
# ──────────────────────────────────────────────────────────────


def compute_validation_report(
    audit_records: List[dict],
    per_prompt_scores: Dict[str, List[float]],  # dim -> list of 0/1 scores
    cost_tracker_data: Optional[Dict] = None,
    auto_time_per_prompt: float = 10.0,
    human_time_per_label: float = 30.0,
) -> Dict:
    """Generate a complete measurement validation report.

    Combines:
        - RQ1: Human vs. auto agreement
        - RQ2: Factors affecting agreement (dimension, attack type, model)
        - RQ3: Dataset size needed for stable scores (via jackknife)
        - RQ4: Cost analysis

    Args:
        audit_records: List of audit records with human/auto labels.
        per_prompt_scores: Dict mapping dimension -> list of 0/1 scores.
        cost_tracker_data: Optional dict from CostTracker.summary().
        auto_time_per_prompt: Seconds per auto inference.
        human_time_per_label: Seconds per human label.

    Returns:
        Comprehensive validation report dict.
    """
    from src.stats import compute_jackknife_stability, compute_dataset_size_sensitivity

    labelled = [r for r in audit_records if r.get("human_label") is not None]
    n_total = len(audit_records)
    n_labelled = len(labelled)

    report = {
        "meta": {
            "total_audit_records": n_total,
            "labelled_records": n_labelled,
            "pct_labelled": round(n_labelled / n_total * 100, 1) if n_total > 0 else 0,
        },
    }

    # RQ1: Agreement
    report["rq1_agreement"] = compute_error_analysis(labelled) if labelled else {}

    # RQ2: Factors
    if labelled:
        report["rq2_factors"] = {
            "by_dimension": _group_agreement(labelled, "dimension"),
            "by_attack_type": _group_agreement(labelled, "attack_type"),
            "by_model": _group_agreement(labelled, "model"),
        }
    else:
        report["rq2_factors"] = {}

    # RQ3: Dataset size stability
    if per_prompt_scores:
        dim_stability = {}
        for dim, scores in per_prompt_scores.items():
            if len(scores) >= 5:
                jackknife = compute_jackknife_stability(scores, n_remove=1)
                size_sens = compute_dataset_size_sensitivity(
                    n_range=[10, 15, 20, 25, 30, 35, 50, 75, 100],
                    base_scores=scores,
                )
                dim_stability[dim] = {
                    "jackknife_stability": jackknife,
                    "dataset_size_sensitivity": size_sens,
                }
        report["rq3_dataset_stability"] = dim_stability
    else:
        report["rq3_dataset_stability"] = {}

    # RQ4: Cost
    cost = compute_measurement_budget(
        auto_time_per_prompt_seconds=auto_time_per_prompt,
        human_time_per_label_seconds=human_time_per_label,
        num_prompts=105,
        num_models=2,
    )
    if cost_tracker_data:
        cost["measured_times"] = cost_tracker_data
    report["rq4_cost"] = cost

    return report
