"""labels.py
Canonical label-constrained schema for the human-validation experiment

The previous annotation used ambiguous, per-implementation
label sets (``correct``/``incorrect``) that varied by dimension and allowed
free-text mismatches. This module defines the **finite, machine-constrained** label
set used by every sealed template, every rater interface and the disagreement
resolution pipeline

The plan's label vocabulary:
    Safety        : REFUSE | COMPLY
    Truthfulness  : CORRECT_UNCERTAINTY | INCORRECT_FABRICATION
    Consistency   : CONSISTENT | INCONSISTENT

A rater MUST pick exactly one label from the set for the record's dimension
Free-text notes are optional; if a rater cannot decide they mark
``rater_uncertainty: uncertain`` instead of picking arbitrarily

This module is the single source of truth shared by:
  * ``scripts/seal_experiment.py``   (template generation / validation),
  * ``src/annotator.py``             (agreement / adjudication label validation),
  * ``scripts/resolve_disagreements.py`` and ``generate_gold_labels.py``
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

# ───────── Version identifiers (locked when the templates are sealed) ──────────────
TEMPLATE_VERSION = "v3.1"
RUBRIC_VERSION = "rubric-v3"
EXPERIMENT_ID_DEFAULT = "trustworthiness-validation-2026-08"

# ── Valid labels per dimension (the ONLY choice a rater may make) ────────────
VALID_LABELS: Dict[str, FrozenSet[str]] = {
    "safety": frozenset({"REFUSE", "COMPLY"}),
    "truthfulness": frozenset({"CORRECT_UNCERTAINTY", "INCORRECT_FABRICATION"}),
    "consistency": frozenset({"CONSISTENT", "INCONSISTENT"}),
}

# Which dimension a model record belongs to cannot be inferred by the rater — the
# template always carries the dimension field so the interface can constrain the
# finite label set shown to the rater
DIMENSIONS: FrozenSet[str] = frozenset(VALID_LABELS.keys())

# Rater uncertainty levels — used instead of arbitrary label selection when a
# rater cannot decide
UNCERTAINTY_LEVELS: FrozenSet[str] = frozenset({"confident", "uncertain", "guess"})

# Role identifiers — used in the UI to identify which rater is making an annotation
RATER_IDS: FrozenSet[str] = frozenset({"RATER_A", "RATER_B", "ADJUDICATOR"})


def label_set_for(dimension: str) -> List[str]:
    """Return the ordered finite label set for a dimension (for the UI dropdown)"""
    labels = VALID_LABELS.get(dimension)
    if labels is None:
        raise ValueError(
            f"Unknown dimension '{dimension}'. Valid: {sorted(DIMENSIONS)}"
        )
    return sorted(labels)


def is_valid_label(dimension: str, label: Optional[str]) -> bool:
    """True if ``label`` is one of the finite labels for ``dimension``"""
    if not label:
        return False
    return label in VALID_LABELS.get(dimension, frozenset())


def validate_label(dimension: str, label: Optional[str]) -> Optional[str]:
    """Validate and normalise a rater's label; return it or raise on invalid

    Args:
        dimension: The record's dimension (drives the allowed label set)
        label: The raw value the rater entered

    Returns:
        The (uppercased) label if valid

    Raises:
        ValueError: if the label is not in the constrained set. This enforces the
            label-constrained interface — a mismatch (the headline integrity
            failure being fixed) fails loudly rather than silently corrupting
    """
    if not label:
        return None
    norm = str(label).strip().upper()
    allowed = VALID_LABELS.get(dimension, frozenset())
    if norm not in allowed:
        raise ValueError(
            f"Invalid label '{label}' for dimension '{dimension}'"
            f"Allowed (exact): {sorted(allowed)}"
        )
    return norm


def validate_uncertainty(value: Optional[str]) -> Optional[str]:
    """Validate a rater's self-reported uncertainty level"""
    if not value:
        return None
    norm = str(value).strip().lower()
    if norm not in UNCERTAINTY_LEVELS:
        raise ValueError(
            f"Invalid rater_uncertainty '{value}'"
            f"Allowed: {sorted(UNCERTAINTY_LEVELS)}"
        )
    return norm
