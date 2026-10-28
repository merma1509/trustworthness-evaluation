#!/usr/bin/env python3
"""report_agreement.py — final agreement figures

Computes the held-out agreement statistics that go into the paper:

  1. Gold (adjudicated human label) VS auto-scorer label:
        overall + per-dimension agreement rate, Cohen's Kappa (+bootstrap CI),
        weighted kappa, and per-label precision/recall/F1.
  2. Inter-rater agreement between RATER_A and RATER_B:
        agreement rate + Cohen's Kappa per dimension.

Both inputs use the SAME constrained label vocabulary (REFUSE/COMPLY, ...), so
gold-vs-auto comparison is an exact string match with no mapping ambiguity.

Usage:
    python3 scripts/report_agreement.py \
        --gold experiment/gold/gold_labels.jsonl \
        --rater-a experiment/annotations/{exp}_RATER_A_heldout.jsonl \
        --rater-b experiment/annotations/{exp}_RATER_B_heldout.jsonl \
        --out experiment/reports/agreement_report.json \
        --with-ci
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agreement import compute_agreement, kappa_bootstrap_ci  # noqa: E402
from src.audit_log import log_action  # noqa: E402
from src.stats import compute_confidence_intervals  # noqa: E402


def _load_gold(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _load_auto_labels(gold_path: Path) -> List[dict]:
    """Load the sealed auto labels co-located with the gold file.

    The gold labels live at ``experiment/gold/gold_labels.jsonl`` while the
    sealed auto labels live at ``experiment/sealed/labels/auto_labels.jsonl``.
    We derive the path relative to the gold file so group-agreement
    block can map each gold K-record back to its (group_id, model_id) — the
    raw consistency outputs don't carry the K-keys, only the sealed labels do.
    """
    sealed = gold_path.parent.parent / "sealed" / "labels" / "auto_labels.jsonl"
    if not sealed.exists():
        return []
    return _load_gold(sealed)


def _per_dimension(records: List[dict], human_key: str, auto_key: str):
    """Compute agreement per dimension from gold records."""
    by_dim: Dict[str, List[tuple]] = defaultdict(list)
    for r in records:
        h = r.get(human_key)
        a = r.get(auto_key)
        if h is not None and a is not None:
            by_dim[r.get("dimension", "unknown")].append((h, a))
    out = {}
    for dim in sorted(by_dim):
        hs = [p[0] for p in by_dim[dim]]
        as_ = [p[1] for p in by_dim[dim]]
        out[dim] = compute_agreement(hs, as_)
        out[dim]["n"] = len(hs)
    return out


def _consistency_group_agreement(
    gold_records: List[dict],
    auto_labels: List[dict],
    with_ci: bool = False,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Raw group-level agreement for consistency, alongside the degenerate κ.

    a Cohen's Kappa over a single observed category (every
    held-out consistency group was judged CONSISTENT) is tautological — κ=1.0
    carries no evidential weight. This block therefore reports consistency as
    **raw group agreement** (cluster-level) with a bootstrap CI,
    plus an explicit single-class caveat so downstream consumers never read the
    degenerate κ as proof of trustworthy consistency.

    Unit of analysis: the **model-scoped group** ``(group_id, model_id)``.
    A group is CONSISTENT only if every covered record is CONSISTENT.
    Singletons / benign controls are excluded (they never cast a scored group
    verdict), matching ``_extract_per_prompt_scores`` and ``evaluate_consistency``.

    Args:
        gold_records: Adjudicated human gold labels (with ``dimension``,
            ``internal_key``, ``gold_human_label``).
        auto_labels: Sealed auto labels (``auto_labels.jsonl``) that map each
            K-record to its raw ``prompt_id``/``group_id``/``model_id``.
        with_ci: Whether to bootstrap-CI the group agreement rate.
        n_bootstrap: Bootstrap iterations (default 10 000).
        seed: RNG seed for reproducibility (default 42).

    Returns:
        Dict with ``agreement_rate``, ``n_groups``, ``n_consistent_groups``,
        ``per_group`` breakdown, ``agreement_ci`` (optional), and a
        ``single_class_caveat``.
    """
    # 1) Map each record key -> (group_id, model_id) from the sealed auto labels.
    key_to_group: Dict[str, tuple] = {}
    for rec in auto_labels:
        if rec.get("dimension") != "consistency":
            continue
        gid = rec.get("group_id")
        if not gid:
            continue
        key = rec.get("internal_key") or rec.get("prompt_id")
        if key:
            key_to_group[key] = (gid, rec.get("model_id") or "")

    # 2) Collect all gold-consistency records covered by a known group.
    #    A group is consistent only if EVERY covered record is CONSISTENT.
    group_gold: Dict[tuple, bool] = {}
    group_n_records: Dict[tuple, int] = defaultdict(int)
    for rec in gold_records:
        if rec.get("dimension") != "consistency":
            continue
        g = key_to_group.get(rec.get("internal_key", ""))
        if not g:
            continue
        group_n_records[g] += 1
        # Explicitly register the group so it is counted even when consistent
        # (a defaultdict(lambda: True) never materialises all-True keys).
        if g not in group_gold:
            group_gold[g] = True
        if (rec.get("gold_human_label") or "").upper() != "CONSISTENT":
            group_gold[g] = False

    # 3) Group the covered groups by their auto-scorer verdict.
    #    The auto verdict comes from the sealed auto labels: a group is
    #    auto-consistent iff every sealed record in it is labelled CONSISTENT.
    group_auto: Dict[tuple, bool] = {}
    for rec in auto_labels:
        if rec.get("dimension") != "consistency":
            continue
        key = rec.get("internal_key") or rec.get("prompt_id")
        g = key_to_group.get(key or "")
        if not g or g not in group_gold:
            continue
        # Explicitly materialise the group key (as above).
        if g not in group_auto:
            group_auto[g] = True
        if (rec.get("auto_label") or "").upper() != "CONSISTENT":
            group_auto[g] = False

    groups = sorted(set(group_gold) | set(group_auto))
    pairs: List[tuple] = []
    per_group = []
    n_consistent_groups = 0
    for g in groups:
        human = group_gold.get(g, True)
        auto = group_auto.get(g, True)
        pairs.append((human, auto))
        if human == auto and human is True:
            n_consistent_groups += 1
        per_group.append(
            {
                "group_id": g[0],
                "model_id": g[1],
                "n_records": group_n_records.get(g, 0),
                "human_verdict": "CONSISTENT" if human else "INCONSISTENT",
                "auto_verdict": "CONSISTENT" if auto else "INCONSISTENT",
                "agree": human == auto,
            }
        )

    n = len(pairs)
    agreement_rate = (n_consistent_groups / n) if n else 0.0
    agreement_ci = None
    if with_ci and n:
        # Cluster the verdicts into 0/1 (1 == group consistent).
        trial_scores = [1.0 if p[0] and p[1] else 0.0 for p in pairs]
        agreement_ci = compute_confidence_intervals(
            trial_scores, n_bootstrap=n_bootstrap, ci=0.95, seed=seed
        )

    return {
        "units": "model-scoped cluster (group_id, model_id)",
        "n_groups": n,
        "n_consistent_groups": n_consistent_groups,
        "agreement_rate": round(agreement_rate, 4),
        "agreement_ci": agreement_ci,
        "per_group": per_group,
        "single_class_caveat": (
            "All covered gold consistency groups were judged CONSISTENT, so "
            "Cohen's Kappa is exactly 1.0 by construction (a single observed "
            "category makes κ tautological). The group-agreement rate above is "
            "the meaningful, non-degenerate statistic; treat the κ=1.0 from "
            "the per-dimension block as a sample-configuration artifact, not "
            "as evidence of trustworthy consistency. Real INCONSISTENT probes "
            "are needed for a defensible 2-class measure."
        ),
    }


def gold_vs_auto(gold_records: List[dict], with_ci: bool) -> Dict:
    """Gold (human-adjudicated) vs auto label agreement."""
    overall = compute_agreement(
        [r["gold_human_label"] for r in gold_records],
        [r["auto_label"] for r in gold_records],
        with_ci=with_ci,
    )
    per_dim = _per_dimension(gold_records, "gold_human_label", "auto_label")

    result = {
        "n_records": len(gold_records),
        "overall": overall,
        "per_dimension": per_dim,
    }
    # annotate the degenerate single-class consistency κ so no
    # consumer reads "κ=1.0 ⇒ trustworthy consistency". A κ computed over a
    # single observed category is a tautology.
    cons_block = result["per_dimension"].get("consistency")
    if isinstance(cons_block, dict):
        cons_block["single_class_caveat"] = (
            "All held-out consistency records were judged CONSISTENT, so "
            "Cohen's Kappa is exactly 1.0 by construction (a single observed "
            "category makes κ tautological). Report consistency as raw "
            "group-agreement with CIs — see 'consistency_group_agreement' — "
            "and treat this κ as a sample-configuration artifact, not as "
            "evidence of trustworthy consistency."
        )
    result["overall"]["confusion_axes"] = {"rows": "auto", "cols": "gold_human"}
    # Spell out the confusion-matrix orientation so consumers (dashboard, report
    # text, verify scripts) cannot read it backwards.  Built in
    # src/agreement._build_confusion as {auto_label: {human_label: count}},
    # i.e. ROWS = auto-scorer label, COLS = gold human label.
    result["confusion_axes"] = {"rows": "auto", "cols": "gold_human"}
    for dim_block in result["per_dimension"].values():
        if isinstance(dim_block, dict):
            dim_block["confusion_axes"] = {"rows": "auto", "cols": "gold_human"}
    if with_ci:
        hs = [r for r in gold_records if r.get("gold_human_label") and r.get("auto_label")]
        result["overall"]["kappa_ci"] = kappa_bootstrap_ci(
            [r["gold_human_label"] for r in hs],
            [r["auto_label"] for r in hs],
        )
    return result


def inter_rater(a_path: Path, b_path: Path) -> Dict:
    """Inter-rater agreement between RATER_A and RATER_B."""
    def _load(path: Path) -> Dict[str, tuple]:
        out = {}
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            dim = rec.get("prompt_record", {}).get("dimension", "")
            ann = rec.get("annotations") or {}
            key = {
                "safety": "safety_label",
                "truthfulness": "truthfulness_label",
                "consistency": "consistency_label",
            }.get(dim)
            label = ann.get(key, "") if key else ""
            out[rec.get("internal_key", "")] = (dim, label or None)
        return out

    a_map = _load(a_path)
    b_map = _load(b_path)

    by_dim: Dict[str, List[tuple]] = defaultdict(list)
    for k, (a_dim, la) in a_map.items():
        if k not in b_map:
            continue
        b_dim, lb = b_map[k]
        if la is None or lb is None:
            continue  # uncertain / not labelled don't cast a vote
        dim = a_dim or b_dim or "unknown"
        by_dim[dim].append((la, lb))

    result = {}
    for dim in sorted(by_dim):
        hs = [p[0] for p in by_dim[dim]]
        as_ = [p[1] for p in by_dim[dim]]
        result[dim] = compute_agreement(hs, as_)
        result[dim]["n"] = len(hs)
        # Rows = RATER_A, cols = RATER_B (built as {A: {B: count}}).
        result[dim]["confusion_axes"] = {"rows": "rater_a", "cols": "rater_b"}
    return result


def _load_resolution_coverage(resolutions_path) -> Dict:
    """Read the held-out disagreement-resolution file to expose how many
    records were actually scored vs left unresolved.

    (report transparency): the agreement figures are computed over
    ``n_scored`` gold labels; the records that no one could adjudicate
    (``unresolvable``) are silently dropped from gold. This block makes that
    leak explicit so the report can never overstate adjudication coverage.
    """
    if not resolutions_path:
        return {}
    resolutions_path = Path(resolutions_path)
    if not resolutions_path.exists():
        return {}
    data = json.loads(resolutions_path.read_text())
    outcomes = data.get("outcomes") or []
    unresolved = sorted(
        o.get("record_key")
        for o in outcomes
        if o.get("outcome") == "unresolvable"
    )
    return {
        "n_recorded": data.get("n_recorded"),
        "n_scored": data.get("n_scored"),
        "n_unresolvable": data.get("n_unresolvable"),
        "n_excluded_response_invalid": data.get("n_excluded_response_invalid"),
        "outcome_counts": data.get("outcome_counts"),
        "unresolved_keys": unresolved,
        "note": (
            "Records marked 'unresolvable' were excluded from gold: the "
            "adjudicator could not cast a valid vote. They are NOT part of "
            "the n above and are listed in full (blind protocol: keys only, "
            "no prompt text)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--rater-a", required=True)
    parser.add_argument("--rater-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resolutions",
        default=None,
        help="Path to the held-out disagreement-resolutions JSON (for "
             "reporting n_recorded/n_scored/n_unresolvable).",
    )
    parser.add_argument("--with-ci", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"  ✗ Gold file not found: {gold_path}")
        return 1
    gold = _load_gold(gold_path)
    auto_labels = _load_auto_labels(gold_path)

    report = {
        "gold_vs_auto": gold_vs_auto(gold, args.with_ci),
        "consistency_group_agreement": _consistency_group_agreement(
            gold, auto_labels, with_ci=args.with_ci
        ),
        "inter_rater_A_vs_B": inter_rater(Path(args.rater_a), Path(args.rater_b)),
        "resolution_coverage": _load_resolution_coverage(args.resolutions),
        "note": (
            "Labels use the constrained vocabulary (REFUSE/COMPLY, ...). "
            "gold_vs_auto compares the adjudicated human label to the sealed "
            "auto-scorer label over the n_scored gold records; "
            "inter_rater compares RATER_A to RATER_B. "
            "consistency_group_agreement reports raw group-level agreement "
            "for consistency (single-class κ is degenerate). "
            "resolution_coverage exposes any records dropped from gold as "
            "unresolvable."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Human-readable summary ─────────────────────────────────────────────
    gva = report["gold_vs_auto"]
    print("  GOLD vs AUTO (held-out):")
    for dim in ["overall"] + sorted(gva["per_dimension"]):
        s = (gva["overall"] if dim == "overall"
             else gva["per_dimension"].get(dim, {}))
        if not s or "n" not in s:
            continue
        k = s.get("cohens_kappa", 0)
        ci = s.get("kappa_ci")
        ci_txt = (f" CI=[{ci['ci_lower']},{ci['ci_upper']}]" if ci else "")
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={k}{ci_txt}")

    ir = report["inter_rater_A_vs_B"]
    print("  INTER-RATER (A vs B):")
    for dim, s in ir.items():
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={s.get('cohens_kappa')}")

    cga = report.get("consistency_group_agreement", {})
    if cga:
        ci = cga.get("agreement_ci") or {}
        ci_txt = (f" CI=[{ci.get('ci_lower')},{ci.get('ci_upper')}]" if ci else "")
        print(
            "  CONSISTENCY group-level (raw agreement):"
            f" n={cga.get('n_groups'):<4} agreement={cga.get('agreement_rate')}{ci_txt} "
            f"({cga.get('n_consistent_groups')} consistent groups)"
        )
        if cga.get("single_class_caveat"):
            print(f"    caveat: {cga['single_class_caveat']}")

    rc = report.get("resolution_coverage") or {}
    if rc:
        print("  HELD-OUT RESOLUTION COVERAGE:")
        print(f"    recorded={rc.get('n_recorded')} scored={rc.get('n_scored')} "
              f"unresolvable={rc.get('n_unresolvable')}")
        ukeys = rc.get("unresolved_keys") or []
        if ukeys:
            print(f"    unresolved keys ({len(ukeys)}): {ukeys}")

    print(f"\n  -> {out_path}")

    # Audit trail the agreement report is a reproducibility
    # checkpoint — log its inputs and the headline agreement stats.
    gva_overall = report["gold_vs_auto"].get("overall", {})
    log_action(
        log_path=Path("experiment/logs/annotation_log.jsonl"),
        action="report_agreement",
        script="scripts/report_agreement.py",
        args={"--gold": args.gold, "--rater-a": args.rater_a,
              "--rater-b": args.rater_b, "--out": args.out},
        input_paths={"gold": gold_path, "rater_a": Path(args.rater_a),
                     "rater_b": Path(args.rater_b)},
        output_paths={"agreement_report": out_path},
        extra={
            "gold_vs_auto_overall": {
                "n": gva_overall.get("n"),
                "agreement_rate": gva_overall.get("agreement_rate"),
                "cohens_kappa": gva_overall.get("cohens_kappa"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
