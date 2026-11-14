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
  * ``scripts/experiment_status.py`` (annotation readiness / validity gate),
  * ``scripts/resolve_disagreements.py`` and ``generate_gold_labels.py``
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

# ───────── Version identifiers (locked when the templates are sealed) ──────────────
TEMPLATE_VERSION = "v3.1"
RUBRIC_VERSION = "rubric-v3"

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


# ── Unscorable consistency displays ──────────────────────────────────────────
# A consistency verdict compares the responses *inside a group*. ``consistency.py``
# samples a group and de-duplicates identical prompts; when a group collapses to a
# single distinct prompt it is marked ``is_singleton`` and skipped in the score
# (see ``trustscore.py``). Those records were nevertheless emitted into the sealed
# templates, so raters were shown a display with ONE response, had nothing to
# compare, and still returned a verdict.
#
# Such a verdict is not evidence of (in)consistency. Every consumer of annotation
# data — sealing, the adjudicator handoff, resolution and ingestion validation —
# must agree on that, which is why the rule lives here rather than being copied
# into each script.
MIN_COMPARABLE_RESPONSES = 2


def consistency_display_size(rec: Optional[dict]) -> Optional[int]:
    """Number of responses shown in a consistency display.

    Returns ``None`` when the notion does not apply (another dimension) or when
    the size is *unknown*. Saved annotation files drop ``pairs`` entirely, and a
    missing field must not be mistaken for an empty group — treating unknown as
    "zero responses" would exclude every consistency record in a legacy run.
    """
    if not rec:
        return None
    pr = rec.get("prompt_record") or {}
    if (pr.get("dimension") or "").strip() != "consistency":
        return None
    if "pairs" not in rec:
        return None  # unknown, not empty
    return len(rec.get("pairs") or [])


def is_unscorable_consistency(rec: Optional[dict]) -> bool:
    """True when a consistency display has too few responses to compare.

    Unknown display sizes are *not* treated as unscorable (see
    ``consistency_display_size``).
    """
    size = consistency_display_size(rec)
    return size is not None and size < MIN_COMPARABLE_RESPONSES

