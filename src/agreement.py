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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def cohen_kappa(
    confusion: Dict[str, Dict[str, int]],
    weighted: bool = False,
) -> Tuple[Optional[float], float, float]:
    """Compute the standard Cohen's Kappa from a confusion matrix.

    The confusion matrix is indexed ``{auto_label: {human_label: count}}``.

    Args:
        confusion: Nested dict {auto_label: {human_label: count}}.
        weighted: If True, compute quadratic weighted kappa (suitable for
            ordered, multi-class labels). If False, use the unweighted kappa.

    Returns:
        Tuple of ``(kappa, p_observed, p_expected)``.

        ``kappa`` is **None when it is mathematically undefined** — i.e. the
        coefficient carries no information and any number returned here would be
        read as a measurement. Two cases qualify:

        * a single observed category (``p_expected == 0``): agreement is perfect
          by construction, so ``1.0`` would be a tautology, not evidence;
        * ``p_expected == 1`` (all mass on one cell of the marginals): the
          denominator vanishes.

        Neither is a zero. Returning ``0.0`` would claim "chance-level
        agreement" and returning ``1.0`` would claim "perfect agreement"; the
        honest answer is that kappa is not defined, and callers must report it
        as such (see ``docs``/report notes). Note that a genuine disagreement
        distribution can also yield ``kappa == 0.0`` legitimately, so the two
        must be distinguishable.

    Reference:
        Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales.
        Educational and Psychological Measurement, 20(1), 37–46.
    """
    labels = sorted(set(confusion.keys()) | {h for row in confusion.values() for h in row})
    if not labels:
        return None, 0.0, 0.0

    n_categories = len(labels)
    # Build a dense, diagonal-aligned matrix over the union of labels.
    mat = np.zeros((n_categories, n_categories), dtype=float)
    for i, a in enumerate(labels):
        for j, h in enumerate(labels):
            mat[i, j] = confusion.get(a, {}).get(h, 0)

    n_total = mat.sum()
    if n_total == 0:
        return None, 0.0, 0.0

    p_observed = float(np.trace(mat)) / n_total

    if n_categories == 1:
        # Only one category observed on either side: p_expected == 0, so kappa
        # is 0/0. Perfect agreement here is guaranteed by the design, not
        # measured — report it as undefined rather than as kappa == 1.
        return None, p_observed, 0.0

    # Marginals: row (auto) and column (human) observed distributions.
    row_marginal = mat.sum(axis=1) / n_total
    col_marginal = mat.sum(axis=0) / n_total
    # Chance agreement from the marginals (standard Cohen, 1960).
    p_expected = float(np.dot(row_marginal, col_marginal))

    if not weighted:
        # Unweighted (nominal) kappa.
        denominator = 1.0 - p_expected
        if denominator == 0:
            return None, p_observed, p_expected
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
            return None, p_observed, p_expected
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
        (h, a)
        for h, a in zip(human_labels, auto_labels)
        if h is not None and h != "" and a is not None and a != ""
    ]
    n = len(pairs)
    if n == 0:
        return {"kappa": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0, "n_bootstrap": 0}

    rng = np.random.RandomState(random_seed)
    base_confusion = _build_confusion(pairs)
    point_kappa, _, point_p_expected = cohen_kappa(base_confusion, weighted=weighted)

    # ── Undefined kappa: no interval exists ────────────────────────────────
    # A single observed category leaves p_expected == 0, so every resample is
    # also degenerate. A percentile interval over a list of undefined values
    # would be meaningless, so return an explicit empty interval instead of
    # inventing bounds.
    if point_kappa is None:
        return {
            "kappa": None,
            "ci_lower": None,
            "ci_upper": None,
            "n": n,
            "n_bootstrap": n_bootstrap,
            "weighted": weighted,
            "ci_level": ci,
            "method": "undefined",
            "note": (
                "Cohen's kappa is undefined (p_expected == 0: a single observed "
                "category), so no confidence interval exists. Use the raw "
                "agreement rate for this slice."
            ),
        }

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = [pairs[i] for i in idx]
        conf = _build_confusion(sample)
        k, _, _ = cohen_kappa(conf, weighted=weighted)
        # Degenerate resamples carry no kappa value; skip rather than coerce
        # them to 0.0, which would bias the interval downward.
        if k is not None:
            boot_vals.append(k)

    if not boot_vals:
        return {
            "kappa": None,
            "ci_lower": None,
            "ci_upper": None,
            "n": n,
            "n_bootstrap": n_bootstrap,
            "weighted": weighted,
            "ci_level": ci,
            "method": "undefined",
            "note": "Every bootstrap resample was degenerate; kappa undefined.",
        }

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
            f"Label lists must have same length: {len(human_labels)} vs {len(auto_labels)}"
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
    pairs = [(h, a) for h, a in zip(human_labels, auto_labels) if h is not None and h != ""]
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
    categories = sorted({h for h, _ in pairs} | {a for _, a in pairs})
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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

    # Make an undefined kappa explicit instead of leaving a bare null that a
    # downstream reader could mistake for "no agreement".
    if kappa is None:
        out["kappa_note"] = (
            "Cohen's kappa is undefined for this slice: "
            f"p_expected={p_expected} leaves no denominator (single observed "
            "category, or all chance mass on one cell). Observed agreement is "
            f"{out['agreement_rate']}, which is guaranteed by the design rather "
            "than measured. Report the agreement rate, not a kappa."
        )
    elif kappa <= 0.0 and agreement_rate >= 0.7:
        # Kappa paradox (Feinstein & Cicchetti, 1990): when the marginal
        # distributions are highly skewed, high raw agreement can coexist with a
        # kappa at or below chance. Here p_observed == p_expected exactly, so
        # kappa's numerator is zero even though ~9 in 10 records agree.
        #
        # The number is arithmetically correct, but read alone it says "chance
        # level", which is false. Explain the situation and point at the raw
        # agreement rate; never silently substitute a different statistic.
        out["kappa_note"] = (
            "Kappa paradox: agreement is high "
            f"({out['agreement_rate']}) yet kappa is {kappa} because "
            f"p_observed == p_expected == {p_expected}. The marginals are highly "
            "skewed (one label dominates), so chance agreement absorbs almost "
            "all of the observed agreement. The value is arithmetically correct "
            "but should not be read as 'chance-level agreement' — report the "
            "agreement rate alongside it, and obtain more examples of the rare "
            "class for a meaningful kappa."
        )

    elif kappa >= 1.0:
        # Perfect kappa, but check it is not a one-case illusion: if the rarest
        # label appears in only a handful of pairs, "perfect" agreement rests on
        # those few decisions. Report the support so the number is not read as
        # robust evidence of reliability.
        rarest_label, rarest_support = min(
            ((label, sum(1 for h, _ in pairs if h == label)) for label in categories),
            key=lambda kv: kv[1],
        )
        if rarest_support <= 3:
            out["kappa_note"] = (
                f"Perfect kappa rests on very little evidence: the rarest label "
                f"({rarest_label}) appears in only {rarest_support} of {n_valid} "
                "pairs, so the agreement on the rare class is decided by a "
                "handful of records. Treat kappa as provisional until more "
                "examples of that class exist."
            )

    if with_ci:
        out["kappa_ci"] = kappa_bootstrap_ci(
            human_labels, auto_labels, n_bootstrap=n_bootstrap, weighted=False
        )

    return out
