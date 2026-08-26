"""Statistical utilities
Provides: bootstrap confidence intervals (independent, paired, clustered),
          weight sensitivity analysis, jackknife, cost tracking,
          dataset size sensitivity
"""

import time
from typing import Dict, List, Optional

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

        results.append(
            {
                "n": n_target,
                "mean_score": round(float(np.mean(means)), 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "ci_width": ci_width,
            }
        )

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


def estimate_required_sample_size(
    precision: float = 0.05,
    confidence: float = 0.95,
    expected_proportion: float = 0.5,
) -> Dict:
    """Estimate the sample size N needed for a stated precision.

    **Defensible sample-size estimator (Task 6).** Unlike the earlier
    ``compute_dataset_size_sensitivity`` shortcut — which only resampled
    existing scores against a fixed ``< 0.10`` threshold — this computes N to
    reach a *stated* half-width at a *stated* confidence level.

    The unit of analysis is the **independent group** (a unique prompt or a
    consistency group), NOT each raw record. Duplicated prompt texts within a
    consistency group are counted once. See ``docs/scoring-spec-v2.md``.

    Uses the Wald-normal approximation solved for N:

        N = ceil( z^2 * p * (1 - p) / precision^2 )

    Args:
        precision: Desired CI half-width (e.g. 0.05 = +/-5%).
        confidence: Confidence level (0.95 -> z = 1.96).
        expected_proportion: Assumed true proportion (0.5 maximises variance).

    Returns:
        Dict with keys 'n_required', 'precision', 'confidence', 'z',
        'expected_proportion', and 'note'.
    """
    from scipy.stats import norm

    if not (0 < precision < 1):
        raise ValueError("precision must be in (0, 1)")
    if not (0 < expected_proportion < 1):
        raise ValueError("expected_proportion must be in (0, 1)")

    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2))
    variance = expected_proportion * (1.0 - expected_proportion)
    n_required = int(np.ceil((z**2) * variance / (precision**2)))

    return {
        "n_required": n_required,
        "precision": precision,
        "confidence": confidence,
        "z": round(z, 4),
        "expected_proportion": expected_proportion,
        "note": (
            f"Independent units needed for +/-{precision:.0%} half-width at "
            f"{confidence:.0%} confidence (p={expected_proportion}): ~{n_required}. "
            "Units are unique prompts / consistency groups, not raw records."
        ),
    }


def compute_confidence_intervals(
    scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> Dict:
    """Compute bootstrap confidence intervals for a list of scores.

    Handles degenerate (extreme) outcomes specially: when every score is the
    same (all 0 or all 1, e.g. a model that is perfect on every consistency
    group), a naive bootstrap collapses to a degenerate ``[x, x]`` interval
    with zero width — uninformative and misleading. In that case a Beta
    posterior (Jeffreys prior, ``Beta(0.5, 0.5)``) is used so the CI still
    conveys the uncertainty inherent in a finite sample (the Rule of Three
    lower bound for a perfect score). See ``docs/scoring-spec-v2.md``.

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
    n_success = int(sum(1 for s in scores if s >= 0.5))
    mean = float(np.mean(scores))

    alpha = (1.0 - ci) / 2.0

    # ── Extreme outcomes: all successes or all failures ─────────────
    # Bootstrap of identical trials yields a degenerate [x, x] CI. Instead use
    # a Beta posterior so the CI reflects finite-sample uncertainty.
    if n_success == n or n_success == 0:
        p_hat = n_success / n
        # Jeffreys prior Beta(0.5, 0.5); posterior Beta(p+n_success+.5, ...)
        a_post = n_success + 0.5
        b_post = (n - n_success) + 0.5
        from scipy.stats import beta

        ci_lower = float(beta.ppf(alpha, a_post, b_post))
        ci_upper = float(beta.ppf(1.0 - alpha, a_post, b_post))
        return {
            "mean": round(mean, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n": n,
            "method": "beta_posterior",
            "note": (
                f"Extreme outcome ({int(p_hat * n)}/{n}); bootstrap degenerate, "
                "used Beta posterior (Jeffreys prior) instead."
            ),
        }

    # ── Standard bootstrap ──────────────────────────────────────────
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    ci_lower = np.percentile(means, alpha * 100)
    ci_upper = np.percentile(means, (1.0 - alpha) * 100)

    return {
        "mean": round(mean, 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "n": n,
        "method": "bootstrap",
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
    common_groups = sorted(set(model1_group_results.keys()) & set(model2_group_results.keys()))
    if len(common_groups) == 0:
        raise ValueError(
            "No common groups between the two models. Cannot compute paired / clustered bootstrap."
        )

    n = len(common_groups)

    # Per-group differences
    differences = [
        (1 if model1_group_results[gid] else 0) - (1 if model2_group_results[gid] else 0)
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


# ──────────────────────────────────────────────────────────────
# Ranking stability (bootstrap probability of ranking flip)
# ──────────────────────────────────────────────────────────────
def compute_ranking_stability(
    model1_dim_scores: Dict[str, float],
    model2_dim_scores: Dict[str, float],
    weight_configs: List[Dict] = None,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict:
    """Bootstrap the probability that model1 beats model2 under each weight config.

    Replaces the qualitative "Llama wins under all weight configs" claim with a
    quantitative probability of each model winning (or a tie) under each
    configuration, given the per-dimension point scores and their uncertainty.

    The dimension scores are treated as fixed point estimates; the bootstrap
    perturbs them around their sampling uncertainty (approximated by a Beta-
    shaped spread derived from a Jeffreys prior over the observed score). This
    yields a conservative probability of ranking flip across weightings.

    Args:
        model1_dim_scores: Dict with keys 'safety','truthfulness','consistency'
            giving each model's point score (0-1) for each dimension.
        model2_dim_scores: Same for model 2.
        weight_configs: List of dicts with 'name','w_s','w_t','w_c'. Defaults to
            DEFAULT_WEIGHT_CONFIGS.
        n_bootstrap: Number of bootstrap iterations per config.
        ci: Confidence level for the reported flip-probability CI.
        random_seed: RNG seed for reproducibility.

    Returns:
        Dict with keys:
            'model_wins': {'model1': pct, 'model2': pct, 'tie': pct}
            'per_config': list of per-config results with 'flip_probability',
                          'model1_wins_pct', 'model2_wins_pct', 'tie_pct'
            'n_bootstrap': number of bootstrap iterations per config
            'interpretation': text guide to reading flip_probability
    """
    import numpy as np

    if weight_configs is None:
        weight_configs = DEFAULT_WEIGHT_CONFIGS

    rng = np.random.RandomState(random_seed)

    def _resample_score(score: float) -> float:
        """Sample a plausible alternative for a point score in (0,1).

        Uses a Beta around the observed score as a surrogate for sampling
        uncertainty (a degenerate point estimate with no variance would give
        trivially stable / unstable results).
        """
        # Jeffreys-prior-like spread: scale alpha by score so we don't blow
        # past [0,1] but still capture meaningful variance.
        if score <= 0.0:
            return float(rng.beta(0.5, 10.0))
        if score >= 1.0:
            return float(rng.beta(10.0, 0.5))
        scale = 8.0  # lower = more spread
        a = max(scale * score, 0.1)
        b = max(scale * (1.0 - score), 0.1)
        return float(np.clip(rng.beta(a, b), 0.0, 1.0))

    overall = {"model1": 0, "model2": 0, "tie": 0}
    per_config = []

    for config in weight_configs:
        w_s, w_t, w_c = config["w_s"], config["w_t"], config["w_c"]
        wins1 = wins2 = ties = 0
        for _ in range(n_bootstrap):
            s1 = w_s * _resample_score(model1_dim_scores["safety"])
            t1 = w_t * _resample_score(model1_dim_scores["truthfulness"])
            c1 = w_c * _resample_score(model1_dim_scores["consistency"])
            s2 = w_s * _resample_score(model2_dim_scores["safety"])
            t2 = w_t * _resample_score(model2_dim_scores["truthfulness"])
            c2 = w_c * _resample_score(model2_dim_scores["consistency"])

            score1 = s1 + t1 + c1
            score2 = s2 + t2 + c2
            if score1 > score2:
                wins1 += 1
            elif score2 > score1:
                wins2 += 1
            else:
                ties += 1

        overall["model1"] += wins1
        overall["model2"] += wins2
        overall["tie"] += ties

        n_cfg = n_bootstrap
        flip_prob = wins2 / n_cfg  # prob model2 (model2) outscores model1
        per_config.append(
            {
                "name": config["name"],
                "w_safety": config["w_s"],
                "w_truthfulness": config["w_t"],
                "w_consistency": config["w_c"],
                "model1_wins_pct": round(wins1 / n_cfg * 100, 1),
                "model2_wins_pct": round(wins2 / n_cfg * 100, 1),
                "tie_pct": round(ties / n_cfg * 100, 1),
                "flip_probability": round(flip_prob, 4),
            }
        )

    total = overall["model1"] + overall["model2"] + overall["tie"]
    return {
        "model_wins": {
            "model1_pct": round(overall["model1"] / total * 100, 1),
            "model2_pct": round(overall["model2"] / total * 100, 1),
            "tie_pct": round(overall["tie"] / total * 100, 1),
        },
        "per_config": per_config,
        "n_bootstrap": n_bootstrap,
        "interpretation": (
            "flip_probability = P(model2 outscores model1) under the config; "
            ">0.95 stable model2 win, 0.5-0.9 unstable, <=0.5 model2 loses."
        ),
    }


# ──────────────────────────────────────────────────────────────
# Empirical required sample size
# ──────────────────────────────────────────────────────────────
def compute_required_n_empirically(
    base_scores: List[float],
    target_precision: float = 0.05,
    target_ci: float = 0.95,
    n_bootstrap: int = 500,
    random_seed: int = 42,
) -> Dict:
    """Empirically find the dataset size N needed for a target CI half-width.

    Complements the theoretical Wald estimate (``estimate_required_sample_size``)
    by *measuring* how the bootstrap CI half-width shrinks as dataset size
    grows. Returns the smallest ``n`` whose bootstrap CI width is at most
    ``2 * target_precision``.

    Args:
        base_scores: List of 0/1 scores from the full dataset.
        target_precision: Desired CI half-width (e.g. 0.05 = +/-5%).
        target_ci: Confidence level (default 0.95).
        n_bootstrap: Bootstrap iterations per simulated size.
        random_seed: RNG seed.

    Returns:
        Dict with keys 'n_required', 'target_precision', 'ci_width_at_n',
        'sizes' (the full size-sweep), and 'note'.
    """
    import numpy as np

    rng = np.random.RandomState(random_seed)
    full_n = len(base_scores)
    if full_n == 0:
        return {"n_required": None, "note": "empty dataset"}

    alpha = (1.0 - target_ci) / 2.0
    max_allowed_width = 2 * target_precision

    # Theoretical Wald N — computed up-front so the empirical sweep can be
    # extended to (at least) cover it. Without this, a small dataset (n<=38)
    # would never reach a tight +/-5% CI within a scan bounded at full_n*3
    # (e.g. 105 max for n=35 << the Wald 332 target), returning None.
    theoretical_wald_n = estimate_required_sample_size(
        precision=target_precision, confidence=target_ci
    )["n_required"]

    # Sweep increasing dataset sizes (bootstrap WITH replacement to simulate
    # larger N beyond the observed count via the empirical scores).
    step = max(1, full_n // 12)
    # Scan from 10 up to at least 1.25x the theoretical target, but not below
    # 3x the observed dataset (which is generous for stable-looking CIs).
    upper = max(full_n * 3, int(theoretical_wald_n * 1.25) + 1)
    n_range = list(range(10, upper + 1, step))
    if upper not in n_range:
        n_range.append(upper)
    else:
        n_range.sort()

    sizes = []
    n_required = None
    for n_target in n_range:
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(base_scores, size=n_target, replace=True)
            means.append(float(np.mean(sample)))
        lo = float(np.percentile(means, alpha * 100))
        hi = float(np.percentile(means, (1.0 - alpha) * 100))
        width = hi - lo
        sizes.append(
            {
                "n": n_target,
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "ci_width": round(width, 4),
            }
        )
        if n_required is None and width <= max_allowed_width:
            n_required = n_target

    return {
        "n_required": n_required,
        "target_precision": target_precision,
        "ci_width_at_n": (sizes[-1]["ci_width"] if sizes else None),
        "sizes": sizes,
        "theoretical_wald_n": estimate_required_sample_size(
            precision=target_precision, confidence=target_ci
        )["n_required"],
        "note": (
            f"Empirically determined smallest N reaching +/-{target_precision:.0%} "
            f"CI ({target_ci:.0%}): {n_required}. "
            f"Full dataset has {full_n} units."
        ),
    }
