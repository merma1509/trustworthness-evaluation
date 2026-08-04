"""agreement.py
Computes inter-rater agreement between human labels and auto-scorer labels.
Provides: Cohen's Kappa, agreement rate, per-dimension breakdown, confusion matrix.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


def _cohen_kappa(observed_agree: float, n_categories: int) -> float:
    """Compute Cohen's Kappa from observed agreement and number of categories.

    Args:
        observed_agree: Proportion of observed agreement (0-1).
        n_categories: Number of label categories (e.g. 2 for binary).

    Returns:
        Kappa value (0-1, could be negative if agreement < expected by chance).
    """
    expected_by_chance = 1.0 / n_categories if n_categories > 0 else 0.0
    if expected_by_chance == 1.0:
        return 1.0  # Only one category → perfect agreement by definition

    kappa = (observed_agree - expected_by_chance) / (1.0 - expected_by_chance)
    return round(kappa, 4)


def compute_agreement(
    human_labels: List[str],
    auto_labels: List[str],
) -> Dict:
    """Compute agreement statistics between human and auto labels.

    Args:
        human_labels: List of human-assigned labels (strings).
        auto_labels: List of auto-scorer labels (strings).
            Must be the same length and in corresponding order.

    Returns:
        Dict with keys:
            'n': number of labelled pairs
            'agreement_rate': proportion of exact matches
            'cohens_kappa': Cohen's Kappa (binary: equal to agreement_rate
                            adjusted for chance)
            'confusion_matrix': {auto_label: {human_label: count}}
            'per_label_agreement': {label: {precision, recall, support}}
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

    # Remove None/null entries
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

    # Confusion matrix: auto → human
    confusion = defaultdict(lambda: defaultdict(int))
    for h, a in pairs:
        confusion[a][h] += 1

    # Agreement rate
    matches = sum(1 for h, a in pairs if h == a)
    agreement_rate = matches / n_valid

    # Cohen's Kappa (binary case: 2 categories)
    unique_labels = set(h for h, _ in pairs) | set(a for _, a in pairs)
    categories = sorted(unique_labels)
    n_categories = len(categories)

    kappa = _cohen_kappa(agreement_rate, n_categories)

    # Per-label precision/recall/support
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

    # Convert confusion matrix to plain dict
    confusion_out = {}
    for auto_label, human_counts in sorted(confusion.items()):
        confusion_out[auto_label] = dict(human_counts)

    return {
        "n": n,
        "n_valid_pairs": n_valid,
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "categories": categories,
        "confusion_matrix": confusion_out,
        "per_label_agreement": per_label,
    }


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
    # Group by dimension
    by_dim = defaultdict(list)
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
