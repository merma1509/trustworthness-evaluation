"""agreement.py
Computes inter-rater agreement between human labels and auto-scorer labels.
Provides: Cohen's Kappa, agreement rate, per-dimension breakdown, confusion matrix.

Note on Cohen's Kappa:
    The chance agreement p_e is estimated from the *marginals* of BOTH raters'
    observed distributions, per the standard definition (Cohen, 1960):

        p_o = (1/N) * sum_k C_kk
        p_e = sum_k (row_k / N) * (col_k / N)
        k   = (p_o - p_e) / (1 - p_e)

    An earlier implementation used p_e = 1/n_categories, which is only correct
    when all marginal label frequencies are equal. That simplification produced
    inflated kappa values and has been removed.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def cohen_kappa(
    confusion: Dict[str, Dict[str, int]],
    weighted: bool = False,
) -> Tuple[float, float, float]:
    """Compute the standard Cohen's Kappa from a confusion matrix.

    The confusion matrix is indexed ``{auto_label: {human_label: count}}``.

    Args:
        confusion: Nested dict {auto_label: {human_label: count}}.
        weighted: If True, compute quadratic weighted kappa (suitable for
            ordered, multi-class labels). If False, use the unweighted kappa.

    Returns:
        Tuple of (kappa, p_observed, p_expected).

    Reference:
        Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales.
        Educational and Psychological Measurement, 20(1), 37–46.
    """
    labels = sorted(
        set(confusion.keys()) | {h for row in confusion.values() for h in row}
    )
    if not labels:
        return 0.0, 0.0, 0.0

    n_categories = len(labels)
    # Build a dense, diagonal-aligned matrix over the union of labels.
    mat = np.zeros((n_categories, n_categories), dtype=float)
    for i, a in enumerate(labels):
        for j, h in enumerate(labels):
            mat[i, j] = confusion.get(a, {}).get(h, 0)

    n_total = mat.sum()
    if n_total == 0:
        return 0.0, 0.0, 0.0

    p_observed = float(np.trace(mat)) / n_total

    if n_categories == 1:
        # Only one category -> perfect agreement by definition (kappa = 1).
        return 1.0, p_observed, 0.0

    # Marginals: row (auto) and column (human) observed distributions.
    row_marginal = mat.sum(axis=1) / n_total
    col_marginal = mat.sum(axis=0) / n_total
    # Chance agreement from the marginals (standard Cohen, 1960).
    p_expected = float(np.dot(row_marginal, col_marginal))

    if not weighted:
        # Unweighted (nominal) kappa.
        denominator = 1.0 - p_expected
        if denominator == 0:
            return 0.0, p_observed, p_expected
        kappa = (p_observed - p_expected) / denominator
    else:
        # Quadratic weighted kappa for ordered categories: weights grow
        # quadratically with category distance, so near-misses are penalised
        # less than far-misses.
        indices = np.arange(n_categories)
        weights = (indices[:, None] - indices[None, :]) ** 2
        expected = np.outer(row_marginal, col_marginal) * n_total
        observed_numerator = float((mat * weights).sum()) / n_total
        # Equivalent closed form for quadratic weighted kappa.
        expected_numerator = float((expected * weights).sum()) / n_total
        if expected_numerator == 0:
            return 0.0, p_observed, p_expected
        kappa = 1.0 - (observed_numerator / expected_numerator)

    return round(kappa, 4), round(p_observed, 4), round(p_expected, 4)


def kappa_bootstrap_ci(
    human_labels: Sequence[str],
    auto_labels: Sequence[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    weighted: bool = False,
    random_seed: int = 42,
) -> Dict:
    """Bootstrap confidence interval for Cohen's Kappa.

    Resamples the labelled *records* (pairs) with replacement, recomputes
    kappa on each resample, and returns the percentile interval.

    Args:
        human_labels: Human-assigned labels (one per record).
        auto_labels: Auto-scorer labels (one per record, same order).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        ci: Confidence level (default 0.95).
        weighted: Whether to bootstrap the weighted kappa instead.
        random_seed: RNG seed for reproducibility.

    Returns:
        Dict with keys 'kappa', 'ci_lower', 'ci_upper', 'n', 'n_bootstrap'.
    """
    pairs = [
        (h, a) for h, a in zip(human_labels, auto_labels)
        if h is not None and h != "" and a is not None and a != ""
    ]
    n = len(pairs)
    if n == 0:
        return {"kappa": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0, "n_bootstrap": 0}

    rng = np.random.RandomState(random_seed)
    base_confusion = _build_confusion(pairs)
    point_kappa, _, _ = cohen_kappa(base_confusion, weighted=weighted)

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = [pairs[i] for i in idx]
        conf = _build_confusion(sample)
        k, _, _ = cohen_kappa(conf, weighted=weighted)
        boot_vals.append(k)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_vals, alpha * 100))
    ci_upper = float(np.percentile(boot_vals, (1.0 - alpha) * 100))

    # ── Perfect agreement: bootstrap is degenerate [1.0, 1.0] ───────────
    # When the observed (and every resampled) kappa is 1.0, a percentile
    # bootstrap collapses to a zero-width [1.0, 1.0] interval that misleadingly
    # claims certainty. Finite-sample uncertainty still exists: with n records
    # and (claimed) perfect agreement, the Rule of Three (3/n) bounds how much
    # a larger sample could plausibly disagree. Applying it symmetrically around
    # kappa=1 gives a lower bound of 1 - 3/n, which approaches 1 as n grows but
    # never claims false certainty on a small sample. Mirrors the Beta-posterior
    # treatment of extreme proportions in ``src.stats.compute_confidence_intervals``.
    if point_kappa >= 1.0 or ci_lower >= 1.0:
        rule_of_three_lower = max(0.0, 1.0 - 3.0 / n)
        return {
            "kappa": round(float(point_kappa), 4),
            "ci_lower": round(rule_of_three_lower, 4),
            "ci_upper": 1.0,
            "n": n,
            "n_bootstrap": n_bootstrap,
            "weighted": weighted,
            "ci_level": ci,
            "method": "rule_of_three",
            "note": (
                f"Perfect agreement ({n}/{n}); bootstrap degenerate, "
                f"used Rule of Three lower bound (1 - 3/{n})."
            ),
        }

    return {
        "kappa": round(float(point_kappa), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "weighted": weighted,
        "ci_level": ci,
        "method": "bootstrap",
    }


def _build_confusion(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """Build a {auto_label: {human_label: count}} confusion matrix."""
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for human, auto in pairs:
        confusion[auto][human] += 1
    return {auto: dict(human_counts) for auto, human_counts in sorted(confusion.items())}


def compute_agreement(
    human_labels: List[str],
    auto_labels: List[str],
    with_ci: bool = False,
    n_bootstrap: int = 1000,
) -> Dict:
    """Compute agreement statistics between human and auto labels.

    Args:
        human_labels: List of human-assigned labels (strings).
        auto_labels: List of auto-scorer labels (strings).
            Must be the same length and in corresponding order.
        with_ci: If True, additionally compute a bootstrap CI on kappa.
        n_bootstrap: Number of bootstrap resamples when ``with_ci`` is True.

    Returns:
        Dict with keys:
            'n': number of labelled pairs
            'agreement_rate': proportion of exact matches
            'cohens_kappa': standard Cohen's Kappa (chance-corrected)
            'weighted_kappa': quadratic weighted kappa (for ordered labels)
            'p_observed': observed agreement (0-1)
            'p_expected': chance agreement (0-1)
            'confusion_matrix': {auto_label: {human_label: count}}
            'per_label_agreement': {label: {precision, recall, support}}
            'kappa_ci': (optional) bootstrap CI dict when ``with_ci`` is True
    """
    if len(human_labels) != len(auto_labels):
        raise ValueError(
            f"Label lists must have same length: "
            f"{len(human_labels)} vs {len(auto_labels)}"
        )

    n = len(human_labels)
    if n == 0:
        return {
            "n": 0,
            "agreement_rate": 0.0,
            "cohens_kappa": 0.0,
            "confusion_matrix": {},
            "per_label_agreement": {},
        }

    # Drop unlabelled pairs.
    pairs = [
        (h, a) for h, a in zip(human_labels, auto_labels)
        if h is not None and h != ""
    ]
    n_valid = len(pairs)

    if n_valid == 0:
        return {
            "n": 0,
            "agreement_rate": 0.0,
            "cohens_kappa": 0.0,
            "confusion_matrix": {},
            "per_label_agreement": {},
            "note": "No human labels found. Run annotation first.",
        }

    # Confusion matrix: auto -> human.
    confusion = _build_confusion(pairs)

    # Agreement rate.
    matches = sum(1 for h, a in pairs if h == a)
    agreement_rate = matches / n_valid

    # Standard Cohen's Kappa (unweighted) from the confusion-matrix marginals.
    categories = sorted(
        {h for h, _ in pairs} | {a for _, a in pairs}
    )
    kappa, p_observed, p_expected = cohen_kappa(confusion, weighted=False)
    weighted_kappa, _, _ = cohen_kappa(confusion, weighted=True)

    # Per-label precision/recall/support.
    per_label = {}
    for label in categories:
        auto_count = sum(1 for _, a in pairs if a == label)
        human_count = sum(1 for h, _ in pairs if h == label)

        tp = confusion.get(label, {}).get(label, 0)
        precision = tp / auto_count if auto_count > 0 else 0.0
        recall = tp / human_count if human_count > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auto_count": auto_count,
            "human_count": human_count,
        }

    out = {
        "n": n,
        "n_valid_pairs": n_valid,
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": kappa,
        "weighted_kappa": weighted_kappa,
        "p_observed": p_observed,
        "p_expected": p_expected,
        "categories": categories,
        "confusion_matrix": confusion,
        "per_label_agreement": per_label,
    }

    if with_ci:
        out["kappa_ci"] = kappa_bootstrap_ci(
            human_labels, auto_labels, n_bootstrap=n_bootstrap, weighted=False
        )

    return out


def compute_per_dimension_agreement(
    records: List[dict],
    human_field: str = "human_label",
    auto_field: str = "auto_label",
    dimension_field: str = "dimension",
) -> Dict[str, Dict]:
    """Compute agreement stats separately for each dimension.

    Args:
        records: List of audit record dicts.
        human_field: Key for human label.
        auto_field: Key for auto label.
        dimension_field: Key for dimension name.

    Returns:
        Dict mapping dimension -> agreement stats dict.
    """
    by_dim: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in records:
        dim = r.get(dimension_field, "unknown")
        h = r.get(human_field)
        a = r.get(auto_field)
        if h is not None and a is not None:
            by_dim[dim].append((h, a))

    results = {}
    for dim in sorted(by_dim):
        human_labels = [p[0] for p in by_dim[dim]]
        auto_labels = [p[1] for p in by_dim[dim]]
        results[dim] = compute_agreement(human_labels, auto_labels)

    return results
