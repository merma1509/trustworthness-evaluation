#!/usr/bin/env python3
"""run_blinded_annotation.py
Orchestrates the blinded multi-rater re-annotation protocol (Task 7, WP-D).

Stage 1 — ``prepare``:  emit one *annotation template* per annotator. Each
    template is a copy of a blinded JSONL (prompt+response only, no
    ``auto_label`` / similarity) with empty ``human_label``/``confidence``/
    ``notes`` fields for the annotator to fill.

Stage 2 — ``report``:   given the filled annotation files (one per annotator),
    compute
        a) inter-annotator agreement (pairwise + mean Cohen's κ) *before* any
           comparison to the auto-scorer — this is the gate,
        b) adjudicated gold labels (majority vote / tie-breaker),
        c) paired the gold vs. auto-scorer agreement (the headline κ).

Usage:
    python3 scripts/run_blinded_annotation.py prepare \
        --input results/audit/blinded/blinded_annotation_calibration.jsonl \
        --output results/blinded_work \
        --annotators ann1 ann2

    python3 scripts/run_blinded_annotation.py report \
        --annotations results/blinded_work/ann1.jsonl results/blinded_work/ann2.jsonl \
        --dimension safety \
        --audit results/audit/all_audit.jsonl \
        --output results/audit/inter_annotator_report.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotator import (
    adjudicate,
    annotator_agreement,
    compare_to_auto,
    merge_annotations,
)


# ──────────────────────────────────────────────────────────────
# Stage 1: prepare annotation templates
# ──────────────────────────────────────────────────────────────
def _emit_templates(input_path: Path, out_dir: Path, annotators: List[str]) -> None:
    """Write one annotation template per annotator under ``out_dir``."""
    records = [json.loads(line) for line in input_path.open() if line.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in annotators:
        out_path = out_dir / f"{name}.jsonl"
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for rec in records:
                template = dict(rec)
                # Carve out exactly the fields the annotator may edit.
                template["human_label"] = ""
                template["confidence"] = None
                template["notes"] = ""
                f.write(json.dumps(template, ensure_ascii=False) + "\n")
        print(f"  Wrote annotation template for '{name}': {out_path}")

    manifest = {
        "input": str(input_path),
        "n_records": len(records),
        "annotators": annotators,
        "instructions": (
            "Fill 'human_label' (correct/incorrect for safety & truthfulness, "
            "consistent/inconsistent for consistency), 'confidence' (0-1), "
            "'notes' (optional). Do NOT look at auto_label — it is absent here."
        ),
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to {out_dir / 'manifest.json'}")
    print("  → Edit each template, then run the 'report' stage.")


# ──────────────────────────────────────────────────────────────
# Stage 2: inter-annotator agreement + gold + auto comparison
# ──────────────────────────────────────────────────────────────
def _load_audit(audit_path: Path) -> List[Dict]:
    return [json.loads(line) for line in audit_path.open() if line.strip()]


def _load_ground_truth(ground_truth_path: Optional[Path]) -> Dict[str, Dict]:
    """Load the analyst-only ``anon_id -> ground truth`` mapping.

    The blinded templates replace every real ``audit_id`` / ``prompt_id`` with a
    neutral sequential ``anon_id``. The ground-truth file (see
    ``generate_blinded_annotation.py``) is the single bridge back to the original
    audit record so the gold-vs-auto comparison can re-attach ``auto_label``.

    Args:
        ground_truth_path: Path to ``ground_truth_blinded.json``, or None.

    Returns:
        A dict with two indexes usable by downstream code:
            ``by_anon_id`` : ``{anon_id: [{audit_id, dimension, attack_type, auto_label}]}``
            ``by_audit_id``: ``{audit_id: {dimension, attack_type, auto_label}}``
        An empty dict when no ground-truth file is supplied (non-anonymised flow).
    """
    if ground_truth_path is None or not Path(ground_truth_path).exists():
        return {}
    with Path(ground_truth_path).open() as f:
        doc = json.load(f)
    # Support both the legacy flat form and the new indexed document.
    if "by_anon_id" in doc:
        return doc
    return {"by_anon_id": {k: [v] for k, v in doc.items()}, "by_audit_id": doc}


def _resolve_anon_labels(gold, ground_truth: Dict[str, Dict]):
    """Rewrite adjudicated gold records keyed by ``anon_id`` -> real ``audit_id``.

    The blinded flow annotates by ``anon_id``; the auto-scorer comparison needs
    the original ``audit_id`` to join against ``all_audit.jsonl``. When a ground
    truth map is present, rebind each gold record via it.

    Args:
        gold: List of adjudicated records carrying ``audit_id`` (which actually
            holds the anonymised id when blinded).
        ground_truth: Ground-truth docs with ``by_anon_id`` / ``by_audit_id``.

    Returns:
        A new list with ``audit_id`` replaced by the real id where resolvable.
    """
    if not ground_truth:
        return list(gold)
    by_anon = ground_truth.get("by_anon_id", {})

    resolved = []
    for g in gold:
        # ``audit_id`` in the gold record is actually an anon unit id (e.g. U_0001).
        anon_id = g.get("audit_id")
        entries = by_anon.get(anon_id) or []
        if entries:
            # Multiple models may share a unit; resolve each record separately
            # so no annotated response is lost.
            for e in entries:
                resolved.append({**g, "audit_id": e.get("audit_id", anon_id)})
        else:
            # Non-anonymised flow (or unresolvable): keep as-is.
            resolved.append(g)
    return resolved


# All dimensions we aggregate into a single report. Keeping them in one file
# (rather than one file per dimension) lets the dashboard show the full
# blinded re-validation instead of only the last dimension that ran.
ALL_DIMENSIONS = ("safety", "truthfulness", "consistency")


def _run_single_dimension(files, audit, dimension, tie_breaker, ground_truth=None):
    """Run one dimension of the blinded re-validation.

    Returns a dict with that dimension's inter-annotator agreement,
    adjudication and gold-vs-auto comparison.

    Args:
        files: Filled per-annotator annotation files.
        audit: Loaded audit records (carry auto_label).
        dimension: Dimension name to restrict labels.
        tie_breaker: Annotator name to prefer on ties.
        ground_truth: Optional ``{anon_id: {audit_id, ...}}`` map to rebind
            anonymised gold records to real audit ids.
    """
    print(f"\n  === [{dimension}] ===")

    # a) Inter-annotator agreement (the gate).
    agreement = annotator_agreement(files, dimension=dimension, with_ci=True)
    print(f"  {agreement['report_text']}")
    for pair, stats in agreement["pairwise"].items():
        if stats.get("n"):
            print(
                f"    {pair:<30} n={stats['n']:<3} "
                f"κ={stats['cohens_kappa']:.3f}  ag={stats['agreement_rate']*100:.1f}%"
            )
        else:
            print(f"    {pair:<30} {stats.get('note','')}")

    # b) Adjudication.
    rows, _names = merge_annotations(files, dimension=dimension)
    gold = adjudicate(rows, tie_breaker=tie_breaker)
    n_unresolved = sum(1 for g in gold if g["needs_adjudication"])
    print(
        f"  Adjudicated {len(gold)} records; "
        f"{n_unresolved} need human adjudication (ties)."
    )

    # Rebind anonymised gold -> real audit id before comparing to the auto scorer.
    gold_for_auto = _resolve_anon_labels(gold, ground_truth or {})

    # c) Gold vs. auto-scorer (headline κ).
    comparison = compare_to_auto(audit, gold_for_auto)
    print("  Gold vs. auto-scorer agreement (headline κ):")
    print(
        f"    n={comparison.get('n_valid_pairs', 0)}  "
        f"κ={comparison.get('cohens_kappa', 0):.3f}  "
        f"agreement={comparison.get('agreement_rate', 0)*100:.1f}%"
    )

    return {
        "inter_annotator": agreement,
        "adjudicated": {
            "n_total": len(gold),
            "n_needs_adjudication": n_unresolved,
            "records": gold,
        },
        "auto_comparison": comparison,
        "used_ground_truth": bool(ground_truth),
    }


def _run_report(args) -> int:
    files = [Path(p) for p in args.annotations]
    if len(files) < 2:
        print("  ✗ Need at least 2 annotation files to measure inter-annotator agreement.")
        return 1

    audit = _load_audit(Path(args.audit))
    ground_truth = _load_ground_truth(
        Path(args.ground_truth) if getattr(args, "ground_truth", None) else None
    )

    # Guard against silently writing a misleading (all-zero) report from
    # UNFILLED templates: an empty human_label today reads as "not annotated".
    # If no annotator has actually cast any label, fail loudly instead of
    # emitting a report that would be misread as a validated null result.
    n_annotated = sum(
        1 for f in files for line in f.open() if line.strip()
        and json.loads(line).get("human_label")
    )
    if n_annotated == 0:
        print("  ✗ No filled 'human_label' values found in the annotation files.")
        print("    The templates appear to be UNFILLED. An empty report would be")
        print("    misleading — refusing to write it. Fill human_label first,")
        print("    e.g. 'make blinded-prepare' then edit results/blinded_work/*.jsonl,")
        print("    or 'make blinded-heldout-prepare' for the held-out set.")
        return 2

    # Aggregate every dimension into a single report so the dashboard shows the
    # full blinded re-validation, not just the last dimension that ran.
    if args.dimension and args.dimension != "all":
        dimensions = [args.dimension]
    else:
        dimensions = list(ALL_DIMENSIONS)

    by_dimension = {}
    for dim in dimensions:
        by_dimension[dim] = _run_single_dimension(
            files, audit, dim, args.tie_breaker, ground_truth
        )

    # Overall / pooled view (across all dimensions).
    rows_all, _names_all = merge_annotations(files, dimension=None)
    gold_all = adjudicate(rows_all, tie_breaker=args.tie_breaker)
    gold_all_for_auto = _resolve_anon_labels(gold_all, ground_truth)
    comparison_all = compare_to_auto(audit, gold_all_for_auto)
    n_unresolved_all = sum(1 for g in gold_all if g["needs_adjudication"])

    print("\n  OVERALL (pooled across dimensions):")
    print(f"    n={comparison_all.get('n_valid_pairs', 0)}  "
          f"κ={comparison_all.get('cohens_kappa', 0):.3f}  "
          f"agreement={comparison_all.get('agreement_rate', 0)*100:.1f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "report",
        "used_ground_truth": bool(ground_truth),
        "by_dimension": by_dimension,
        "overall": {
            "inter_annotator": annotator_agreement(files, dimension=None, with_ci=True),
            "adjudicated": {
                "n_total": len(gold_all),
                "n_needs_adjudication": n_unresolved_all,
                "records": gold_all,
            },
            "auto_comparison": comparison_all,
        },
        "note": (
            "First verify inter_annotator.mean_kappa meets your quality gate "
            "before trusting auto_comparison. Per-dimension results are under "
            "`by_dimension`; a pooled overall view is under `overall`."
        ),
    }
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report written to {out_path}")
    return 0

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_prepare = sub.add_parser("prepare", help="Emit per-annotator templates.")
    p_prepare.add_argument("--input", required=True, help="Blinded JSONL (calibration or heldout).")
    p_prepare.add_argument("--output", default="results/blinded_work")
    p_prepare.add_argument("--annotators", nargs="+", required=True, help="Annotator names (file stems).")

    p_report = sub.add_parser("report", help="Compute inter-annotator + gold + auto agreement.")
    p_report.add_argument("--annotations", nargs="+", required=True, help="Filled annotation JSONL files.")
    p_report.add_argument("--dimension", choices=["safety", "truthfulness", "consistency", "all"],
                          default="all", help="Dimension for label validation, or 'all' to "
                          "aggregate every dimension into one report (default: all).")
    p_report.add_argument("--audit", default="results/audit/all_audit.jsonl")
    p_report.add_argument("--ground-truth", default=None,
                          help="Analyst-only ground_truth_blinded.json mapping "
                               "anon_id -> audit_id (for blinded templates).")
    p_report.add_argument("--tie-breaker", default=None, help="Annotator to prefer on ties.")
    p_report.add_argument("--with-ci", action="store_true", help="Include bootstrap CIs.")
    p_report.add_argument("--output", default="results/audit/inter_annotator_report.json")

    args = parser.parse_args()

    if args.stage == "prepare":
        _emit_templates(Path(args.input), Path(args.output), args.annotators)
        return 0
    return _run_report(args)


if __name__ == "__main__":
    sys.exit(main())
