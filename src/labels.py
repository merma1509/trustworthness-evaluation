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


# ── Resolution coverage accounting ───────────────────────────────────────────
# A resolution report partitions every recorded key into exactly one bucket:
# scored / unresolvable / excluded. There is more than one *reason* to exclude a
# record from scoring, and each is reported under its own counter. Every
# consumer that re-checks "nothing was silently dropped" must sum them all —
# an earlier gate only added ``response_invalid`` to the total and therefore
# reported a broken invariant on any run that excluded a singleton consistency
# display (63 recorded vs 49 + 11 + 0 accounted), even though nothing was lost.
#
# The rule is stated once here so the gate, the agreement report and the
# resolution driver cannot drift apart again.
EXCLUSION_COUNTERS: tuple = (
    "n_excluded_response_invalid",
    "n_excluded_unscorable_consistency",
)


def accounted_records(report: Optional[dict]) -> Optional[int]:
    """Total records ``report`` claims to have accounted for, or ``None``.

    Sums the scored records, the explicitly unresolvable ones and every
    exclusion bucket (see ``EXCLUSION_COUNTERS``). Returns ``None`` when the
    report does not expose the three mandatory counters, so a caller can tell
    "not reported" apart from "reported as zero" instead of treating a missing
    field as an empty bucket.
    """
    if not report:
        return None
    mandatory = ("n_scored", "n_unresolvable")
    if any(report.get(k) is None for k in mandatory):
        return None
    total = int(report["n_scored"]) + int(report["n_unresolvable"])
    for counter in EXCLUSION_COUNTERS:
        # A missing exclusion counter means "this run did not use that reason",
        # which is legitimately zero — but only for the optional counters.
        total += int(report.get(counter) or 0)
    return total


def coverage_gaps(report: Optional[dict]) -> List[str]:
    """Human-readable list of coverage inconsistencies in ``report``.

    An empty list means the report accounts for exactly ``n_recorded`` records.
    Each returned string names one broken invariant so the caller can surface a
    precise reason instead of a bare "coverage broken".
    """
    if not report:
        return ["no resolution report"]
    n_recorded = report.get("n_recorded")
    if n_recorded is None:
        return ["resolution report missing 'n_recorded'"]
    total = accounted_records(report)
    if total is None:
        return ["resolution report missing 'n_scored'/'n_unresolvable'"]

    gaps: List[str] = []
    if total != int(n_recorded):
        gaps.append(
            f"accounted({total}) != recorded({n_recorded}) "
            f"[scored={report.get('n_scored')}, "
            f"unresolvable={report.get('n_unresolvable')}, "
            + ", ".join(
                f"{c}={report.get(c) or 0}" for c in EXCLUSION_COUNTERS
            )
            + "]"
        )

    # The per-reason counters must add up to the reported exclusion total, so a
    # future exclusion reason cannot be counted in one place and lost in another.
    reason_counts = report.get("exclusion_reason_counts") or {}
    if reason_counts:
        declared = sum(int(report.get(c) or 0) for c in EXCLUSION_COUNTERS)
        if sum(int(v) for v in reason_counts.values()) != declared:
            gaps.append(
                "exclusion_reason_counts do not sum to the exclusion counters "
                f"({reason_counts} vs {declared})"
            )
    return gaps

