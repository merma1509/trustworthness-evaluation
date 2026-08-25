"""annotator.py
Tools for the blinded multi-rater re-annotation protocol (Task 7, WP-D).

This script assumes that the original audit leaked ``auto_label`` and
similarity fields to the annotator, and that there was no calibration / held-out
split nor any inter-annotator agreement measurement. This module implements the
*measurement* half of that fix:

    1. ``merge_annotations``  – combine per-annotator JSONL files into one table.
    2. ``annotator_agreement`` – compute pairwise and mean Cohen's κ between the
       human annotators **before** any comparison to the auto-scorer.
    3. ``adjudicate``         – resolve disagreements into a single gold label,
       with optional recording of which annotator(s) won.
    4. ``compare_to_auto``    – the *post-adjudication* comparison that was the
       sole source of the old headline κ.

The blinded JSONL files contain prompt+response only (no ``auto_label``, no
similarity fields), so an annotator's labels are independent of the auto-scorer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.agreement import compute_agreement

# Labels the rubric accepts per dimension (see docs/scoring-spec-v2.md).
VALID_LABELS = {
    "safety": ("correct", "incorrect"),
    "truthfulness": ("correct", "incorrect"),
    "consistency": ("consistent", "inconsistent"),
}


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalise a raw annotator value or return ``None`` if unset."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "null", "-", "--", "skip"):
        return None
    return s


def load_annotation_file(path: Path, dimension: Optional[str] = None) -> Dict[str, str]:
    """Load one annotator's JSONL into ``{audit_id: label}``.

    Accepts either records that already carry ``human_label``/``label`` or bare
    ``{audit_id: label}`` pairs. Unset labels are dropped.

    Args:
        path: Path to a single annotator's JSONL file.
        dimension: If given, only keep records whose ``dimension`` field matches
            (records lacking a dimension field are kept unchanged so bare files,
            e.g. ``{audit_id: label}``, still work). Useful when a template mixes
            several dimensions but the report targets one.

    Returns:
        Mapping of ``audit_id`` -> cleaned label.
    """
    labels: Dict[str, str] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # In the blinded (anonymised) flow records carry a neutral ``anon_id``
        # instead of the real ``audit_id``. Accept both so the measurement half
        # (merge/agreement/adjudicate) works without leaking ground truth.
        audit_id = rec.get("audit_id") or rec.get("id") or rec.get("anon_id")
        if not audit_id:
            continue
        if dimension is not None and rec.get("dimension") and rec.get("dimension") != dimension:
            continue
        label = _clean(rec.get("human_label") or rec.get("label") or rec.get("answer"))
        if label:
            labels[str(audit_id)] = label
    return labels


def _validate_labels(labels: Dict[str, str], dimension: Optional[str] = None) -> None:
    """Raise if a label is not one of the rubric's allowed values."""
    if not dimension:
        return
    allowed = set(VALID_LABELS.get(dimension, ()))
    if not allowed:
        return
    bad = sorted(v for v in labels.values() if v not in allowed)
    if bad:
        raise ValueError(
            f"Invalid labels for dimension '{dimension}': {bad}. Allowed: {sorted(allowed)}"
        )


def merge_annotations(
    annotator_files: Sequence[Path],
    dimension: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, str]]:
    """Merge N annotator JSONL files into a per-record table.

    Args:
        annotator_files: One JSONL path per annotator.
        dimension: Optional dimension to validate labels against the rubric.

    Returns:
        (rows, annotator_names) where each row is
        ``{"audit_id": ..., "annotator": name, "label": ..., "confidence": ...}``
        and ``annotator_names`` maps a numeric index to the file's stem.
    """
    rows: List[Dict] = []
    names: Dict[str, str] = {}
    for idx, path in enumerate(annotator_files):
        name = Path(path).stem
        names[str(idx)] = name
        labels = load_annotation_file(path, dimension=dimension)
        _validate_labels(labels, dimension)
        for audit_id, label in labels.items():
            rows.append(
                {
                    "audit_id": audit_id,
                    "annotator": str(idx),
                    "annotator_name": name,
                    "label": label,
                }
            )
    return rows, names


def annotator_agreement(
    annotator_files: Sequence[Path],
    dimension: Optional[str] = None,
    with_ci: bool = False,
) -> Dict:
    """Compute inter-annotator agreement statistics.

    Computes a Cohen's κ for **every pair** of annotators (on the records they
    both labelled) and reports the mean pairwise κ. This is the gate *before*
    comparing any annotator to the auto-scorer.

    Args:
        annotator_files: One JSONL path per annotator (≥2).
        dimension: Optional dimension for label validation.
        with_ci: Whether to include bootstrap CIs.

    Returns:
        Dict with keys: 'method', 'n_annotators', 'pairwise', 'mean_kappa',
        'mean_agreement_rate', 'n_records_common', 'report_text'.

    Raises:
        ValueError: if fewer than 2 annotators are supplied.
    """
    files = list(annotator_files)
    if len(files) < 2:
        raise ValueError("Need at least 2 annotators to measure inter-annotator agreement.")

    tables = {Path(f).stem: load_annotation_file(f, dimension=dimension) for f in files}
    for table in tables.values():
        _validate_labels(table, dimension)

    names = {i: Path(f).stem for i, f in enumerate(files)}
    pairwise = {}
    kappas: List[float] = []
    agreement_rates: List[float] = []

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            name_i, name_j = names[i], names[j]
            table_i, table_j = tables[name_i], tables[name_j]
            common = sorted(set(table_i) & set(table_j))
            if not common:
                pairwise[f"{name_i}__vs__{name_j}"] = {
                    "n": 0,
                    "note": "No common annotated records.",
                }
                continue

            human = [table_i[a] for a in common]
            other = [table_j[a] for a in common]
            ag = compute_agreement(human, other, with_ci=with_ci)
            pairwise[f"{name_i}__vs__{name_j}"] = {
                "n": len(common),
                "cohens_kappa": ag["cohens_kappa"],
                "weighted_kappa": ag["weighted_kappa"],
                "agreement_rate": ag["agreement_rate"],
                "kappa_ci": ag.get("kappa_ci"),
            }
            kappas.append(ag["cohens_kappa"])
            agreement_rates.append(ag["agreement_rate"])

    n_common = sum(p.get("n", 0) for p in pairwise.values())

    mean_kappa = sum(kappas) / len(kappas) if kappas else 0.0
    mean_agreement = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0.0

    report_text = (
        f"Inter-annotator agreement over {len(files)} annotators: "
        f"mean pairwise Cohen's κ = {mean_kappa:.3f}, "
        f"mean agreement = {mean_agreement * 100:.1f}% "
        f"({len(kappas)} pair(s))."
    )

    return {
        "method": "pairwise_cohen_kappa",
        "n_annotators": len(files),
        "annotators": [names[i] for i in range(len(files))],
        "pairwise": pairwise,
        "mean_kappa": round(mean_kappa, 4),
        "mean_agreement_rate": round(mean_agreement, 4),
        "n_pairwise_terms": len(kappas),
        "n_common_annotations_total": n_common,
        "report_text": report_text,
    }


def adjudicate(
    rows: Iterable[Dict],
    tie_breaker: Optional[str] = None,
) -> List[Dict]:
    """Resolve multi-annotator labels into a single gold label per record.

    Majority vote; on a tie an optional named annotator breaks it. Records where
    no annotator left a label are marked ``human_label=None`` (unlabelled) and
    are excluded from downstream agreement.

    Args:
        rows: Rows as produced by :func:`merge_annotations`.
        tie_breaker: Annotator name (file stem) to prefer on 3-way ties.

    Returns:
        One record per unique ``audit_id`` with the resolved ``human_label``.
    """
    # Keep the annotator who cast each vote so tie-breaking is meaningful.
    votes_by_id: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("label"):
            votes_by_id[row["audit_id"]].append((row["annotator_name"], row["label"]))

    resolved: List[Dict] = []
    for audit_id, votes in sorted(votes_by_id.items()):
        from collections import Counter

        counts = Counter(label for _, label in votes)
        winner, top_count = counts.most_common(1)[0]

        if top_count > len(votes) / 2:
            # A strict majority exists -> majority vote wins.
            final_label = winner
            needs_adjudication = False
        elif tie_breaker:
            # Prefer the named tie-breaker's own vote when it is in (shared) lead.
            breaker_vote = next((label for name, label in votes if name == tie_breaker), None)
            if breaker_vote is not None and counts[breaker_vote] == top_count:
                final_label = breaker_vote
                needs_adjudication = False
            else:
                final_label = None
                needs_adjudication = True
        else:
            # Genuine tie (e.g. 1-1 with two annotators, or a 3-way split).
            final_label = None
            needs_adjudication = True

        resolved.append(
            {
                "audit_id": audit_id,
                "human_label": final_label,
                "annotator_count": len(votes),
                "votes": dict(counts),
                "needs_adjudication": needs_adjudication,
            }
        )
    return resolved


def compare_to_auto(
    audit_records: Sequence[Dict],
    gold: Sequence[Dict],
) -> Dict:
    """Compare adjudicated gold labels to the auto-scorer (the headline κ).

    This is the *second* stage of the protocol: it is only meaningful once
    inter-annotator agreement (see :func:`annotator_agreement`) is acceptable.
    It reuses the standard agreement machinery (correct Cohen's κ with CI).

    Args:
        audit_records: The original audit records carrying ``auto_label``.
        gold: Adjudicated records carrying ``human_label`` (see :func:`adjudicate`).

    Returns:
        Dict matching the ``compute_agreement`` schema, restricted to records
        that have both an auto label and a gold human label.
    """
    gold_by_id = {g["audit_id"]: g for g in gold}
    humans: List[str] = []
    autos: List[str] = []
    for rec in audit_records:
        g = gold_by_id.get(rec.get("audit_id"))
        if g is None or not g.get("human_label"):
            continue
        humans.append(g["human_label"])
        autos.append(rec.get("auto_label", ""))
    return compute_agreement(humans, autos, with_ci=True)
