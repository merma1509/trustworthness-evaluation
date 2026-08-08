"""Statistical utilities
Provides: bootstrap confidence intervals (independent, paired, clustered),
          weight sensitivity analysis, jackknife, cost tracking,
          dataset size sensitivity
"""
from typing import Dict, List, Optional
import time
import numpy as np


# ──────────────────────────────────────────────────────────────
# Cost tracking
# ──────────────────────────────────────────────────────────────


class CostTracker:
    """Tracks inference time and cost for evaluation runs.

    Usage:
        tracker = CostTracker()
        tracker.start("dimension")
        # ... run evaluation ...
        tracker.stop("dimension", n_prompts=35)
        print(tracker.summary())
    """

    def __init__(self):
        self._timers: Dict[str, float] = {}
        self._records: List[Dict] = []

    def start(self, label: str):
        self._timers[label] = time.time()

    def stop(self, label: str, n_prompts: int = 1, extra: Optional[Dict] = None) -> Dict:
        elapsed = time.time() - self._timers.pop(label, time.time())
        record = {
            "label": label,
            "elapsed_seconds": round(elapsed, 2),
            "n_prompts": n_prompts,
            "seconds_per_prompt": round(elapsed / n_prompts, 3) if n_prompts > 0 else 0,
        }
        if extra:
            record.update(extra)
        self._records.append(record)
        return record

    def summary(self) -> Dict:
        """Return aggregate cost summary."""
        if not self._records:
            return {"total_seconds": 0, "total_prompts": 0, "records": []}
        total_sec = sum(r["elapsed_seconds"] for r in self._records)
        total_pr = sum(r["n_prompts"] for r in self._records)
        return {
            "total_seconds": round(total_sec, 2),
            "total_minutes": round(total_sec / 60, 2),
            "total_prompts": total_pr,
            "avg_seconds_per_prompt": round(total_sec / total_pr, 3) if total_pr > 0 else 0,
            "records": self._records,
        }

    def merge(self, other: "CostTracker"):
        """Merge another tracker's records into this one."""
        self._records.extend(other._records)


# ──────────────────────────────────────────────────────────────
# Jackknife stability analysis
# ──────────────────────────────────────────────────────────────


def compute_jackknife_stability(
    per_prompt_scores: List[float],
    n_remove: int = 1,
    n_iterations: int = 1000,
    metric: str = "mean",
    random_seed: int = 42,
) -> Dict:
    """Jackknife / leave-k-out stability analysis.

    Repeatedly removes a small subset of prompts, recomputes the score,
    and measures how much it varies.

    Args:
        per_prompt_scores: List of 0/1 scores per prompt.
        n_remove: Number of prompts to remove per iteration (default=1).
        n_iterations: Number of random subsets (default=1000).
        metric: Which statistic to compute on each subset ("mean" or "accuracy").
        random_seed: RNG seed for reproducibility.

    Returns:
        Dict with keys:
            'full_score': score on all data
            'mean_jackknife': mean of jackknife scores
            'std_jackknife': std dev of jackknife scores (stability measure)
            'ci_95': [lower, upper] percentile interval
            'max_decrease': worst-case drop from full score
            'max_increase': worst-case increase from full score
            'n_total': total number of prompts
            'n_remove': number removed per iteration
    """
    rng = np.random.RandomState(random_seed)
    full_score = float(np.mean(per_prompt_scores))
    n = len(per_prompt_scores)

    jackknife_scores = []
    for _ in range(n_iterations):
        # Randomly select n_remove indices to exclude
        keep_idx = rng.choice(n, size=n - n_remove, replace=False)
        subset = [per_prompt_scores[i] for i in keep_idx]
        jackknife_scores.append(float(np.mean(subset)))

    return {
        "full_score": round(full_score, 4),
        "mean_jackknife": round(float(np.mean(jackknife_scores)), 4),
        "std_jackknife": round(float(np.std(jackknife_scores)), 4),
        "ci_95": [
            round(float(np.percentile(jackknife_scores, 2.5)), 4),
            round(float(np.percentile(jackknife_scores, 97.5)), 4),
        ],
        "max_decrease": round(full_score - min(jackknife_scores), 4),
        "max_increase": round(max(jackknife_scores) - full_score, 4),
        "n_total": n,
        "n_remove": n_remove,
    }


# ──────────────────────────────────────────────────────────────
# Dataset size sensitivity
# ──────────────────────────────────────────────────────────────


def compute_dataset_size_sensitivity(
    n_range: List[int],
    base_scores: List[float],
    n_bootstrap: int = 500,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict:
    """Estimate how CI width shrinks as dataset size grows.

    Resamples smaller datasets from base_scores and measures CI width.

    Args:
        n_range: List of dataset sizes to simulate (e.g., [10, 20, 30, 50]).
        base_scores: List of 0/1 scores from the full dataset.
        n_bootstrap: Bootstrap iterations per size.
        ci: Confidence level.
        random_seed: RNG seed.

    Returns:
        Dict with keys:
            'full_score': mean of base_scores
            'full_n': len(base_scores)
            'sizes': list of {n, ci_width, ci_lower, ci_upper, mean_score}
            'estimated_min_n': estimated N for CI width < 0.10
    """
    rng = np.random.RandomState(random_seed)
    full_n = len(base_scores)
    full_score = float(np.mean(base_scores))

    results = []
    for n_target in sorted(n_range):
        if n_target > full_n:
            continue
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(base_scores, size=n_target, replace=False)
            means.append(float(np.mean(sample)))

        alpha = (1.0 - ci) / 2.0
        ci_lower = float(np.percentile(means, alpha * 100))
        ci_upper = float(np.percentile(means, (1.0 - alpha) * 100))
        ci_width = round(ci_upper - ci_lower, 4)

        results.append({
            "n": n_target,
            "mean_score": round(float(np.mean(means)), 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": ci_width,
        })

    # Estimate minimum N for CI width < 0.10 (rule-of-thumb for binary data)
    estimated_min_n = full_n
    for r in reversed(results):
        if r["ci_width"] < 0.10:
            estimated_min_n = r["n"]
        else:
            break

    return {
        "full_score": round(full_score, 4),
        "full_n": full_n,
        "ci_level": ci,
        "sizes": results,
        "estimated_min_n": estimated_min_n,
        "note": (
            f"Estimated minimum N for 95% CI width < 0.10: ~{estimated_min_n} prompts. "
            f"Current dataset: {full_n} prompts."
        ),
    }


def compute_confidence_intervals(
    scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> Dict:
    """Compute bootstrap confidence intervals for a list of scores.

    Args:
        scores: List of individual trial scores (0 or 1)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dict with keys: 'mean', 'ci_lower', 'ci_upper', 'n'
    """
    if len(scores) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    n = len(scores)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    ci_lower = np.percentile(means, alpha * 100)
    ci_upper = np.percentile(means, (1.0 - alpha) * 100)

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": n,
    }


def compute_paired_difference_ci(
    model1_scores: List[float],
    model2_scores: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Dict:
    """Bootstrap confidence interval for the PAIRED difference.

    Resamples **pairs** (keeping the pairing intact) so the within-prompt
    correlation between model responses is preserved.

    Suitable for **safety** and **truthfulness** dimensions where each
    prompt produces exactly one score (0/1) per model.

    Args:
        model1_scores: List of 0/1 scores for model 1 (one per prompt).
        model2_scores: List of 0/1 scores for model 2 (one per prompt).
            Must be the same length and in the same prompt order.
        n_bootstrap: Number of bootstrap iterations (default: 10 000).
        ci: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Dict with keys:
            'mean_difference':  mean(model1) - mean(model2)  (>0 means
                                model1 is better)
            'ci_lower':         lower bound of the CI on the difference
            'ci_upper':         upper bound of the CI on the difference
            'p_value':          one-sided p-value for H0: difference ≤ 0
                                (i.e. model1 is *not* better than model2)
            'n_pairs':          number of paired observations

    Raises:
        ValueError: if the score lists have different lengths.
    """
    if len(model1_scores) != len(model2_scores):
        raise ValueError(
            f"Score lists must be paired (same length), "
            f"got {len(model1_scores)} vs {len(model2_scores)}"
        )

    n = len(model1_scores)
    if n == 0:
        return {
            "mean_difference": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "n_pairs": 0,
        }

    # Per-prompt differences (positive = model1 better on that prompt)
    differences = [m1 - m2 for m1, m2 in zip(model1_scores, model2_scores)]
    mean_diff = float(np.mean(differences))

    # Bootstrap the mean difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        boot_diffs.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    # One-sided p-value: H0 — model1 is NOT better than model2
    # i.e. true difference ≤ 0.  Count bootstrap samples where diff ≤ 0.
    p_value = float(np.mean([d <= 0.0 for d in boot_diffs]))

    return {
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "n_pairs": n,
    }


def compute_clustered_consistency_ci(
    model1_group_results: Dict[str, bool],
    model2_group_results: Dict[str, bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Dict:
    """Clustered bootstrap CI for the difference in consistency scores.

    Consistency is evaluated **at the group level** (each group is a
    cluster of semantically related prompts).  Standard paired bootstrap
    would incorrectly treat each prompt as independent.  Instead we
    resample **whole groups** (clusters) with replacement.

    Args:
        model1_group_results:  Dict mapping group_id -> bool (consistent?).
        model2_group_results:  Same for model 2.
            Only groups present in **both** dicts are used (intersection).
        n_bootstrap: Number of bootstrap iterations (default: 10 000).
        ci: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Dict with keys:
            'mean_difference':  mean_consistent(model1) -
                                mean_consistent(model2)
            'ci_lower':         lower bound of the CI on the difference
            'ci_upper':         upper bound of the CI on the difference
            'p_value':          one-sided p-value for H0: diff ≤ 0
            'n_groups':         number of groups in the intersection

    Raises:
        ValueError: if the group intersection is empty.
    """
    # Intersection of groups (only groups present in both models)
    common_groups = sorted(
        set(model1_group_results.keys()) & set(model2_group_results.keys())
    )
    if len(common_groups) == 0:
        raise ValueError(
            "No common groups between the two models. "
            "Cannot compute paired / clustered bootstrap."
        )

    n = len(common_groups)

    # Per-group differences
    differences = [
        (1 if model1_group_results[gid] else 0)
        - (1 if model2_group_results[gid] else 0)
        for gid in common_groups
    ]
    mean_diff = float(np.mean(differences))

    # Bootstrap groups (clusters) with replacement
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        boot_diffs.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    # One-sided p-value: H0 — model1 is NOT better than model2
    p_value = float(np.mean([d <= 0.0 for d in boot_diffs]))

    return {
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "n_groups": n,
    }


def compute_weight_sensitivity(
    safety_score: float,
    truthfulness_score: float,
    consistency_score: float,
    weight_configs: List[Dict],
) -> List[Dict]:
    """Compute trustworthiness scores under different weight configurations.

    Args:
        safety_score: Safety dimension score (0-1)
        truthfulness_score: Truthfulness dimension score (0-1)
        consistency_score: Consistency dimension score (0-1)
        weight_configs: List of dicts with keys 'name', 'w_s', 'w_t', 'w_c'

    Returns:
        List of dicts with scores under each configuration
    """
    results = []
    for config in weight_configs:
        total = (
            config["w_s"] * safety_score
            + config["w_t"] * truthfulness_score
            + config["w_c"] * consistency_score
        )
        results.append(
            {
                "name": config["name"],
                "w_safety": config["w_s"],
                "w_truthfulness": config["w_t"],
                "w_consistency": config["w_c"],
                "score": round(total, 4),
            }
        )
    return results


# Default weight configurations for sensitivity analysis
DEFAULT_WEIGHT_CONFIGS = [
    {"name": "Baseline (Safety-priority)", "w_s": 0.40, "w_t": 0.35, "w_c": 0.25},
    {"name": "Safety-heavy", "w_s": 0.60, "w_t": 0.25, "w_c": 0.15},
    {"name": "Balanced", "w_s": 0.33, "w_t": 0.33, "w_c": 0.34},
    {"name": "FPR-heavy", "w_s": 0.25, "w_t": 0.50, "w_c": 0.25},
    {"name": "Consistency-heavy", "w_s": 0.20, "w_t": 0.40, "w_c": 0.40},
]
