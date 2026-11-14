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


def _load_auto_labels(gold_path: Path, override: Optional[str] = None) -> List[dict]:
    """Load the sealed auto labels used for display identity and test-retest.

    The gold labels live at ``experiment/gold/gold_labels.jsonl`` while the
    sealed auto labels live at ``experiment/sealed/labels/auto_labels.jsonl``.
    Deriving the path relative to the gold file works only when gold sits in
    its canonical location: a rerun that writes gold to a temporary directory
    made this lookup fail, which silently emptied the display-level and
    test-retest blocks instead of raising. Callers that relocate gold must pass
    ``override`` (CLI: ``--auto-labels``).
    """
    if override:
        path = Path(override)
        return _load_gold(path) if path.exists() else []
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


# ── Display-level accounting for consistency ────────────────────────────────
# A *display* is what a rater actually sees: (display_model, sorted prompts).
# The sealer emitted one template record per raw-output record, so the same
# display appears under several internal_keys and was rated several times.
#
# Counting agreement per internal_key therefore double-counts: duplicates always
# agree with themselves, so they inflate n (narrowing every CI) while adding no
# information. This block recomputes consistency on display units and keeps the
# duplicate ratings, which are a free test-retest reliability measurement.
#
# Only consistency uses displays; safety/truthfulness are flat single-response
# records where one internal_key is exactly one observation.

def _display_key(rec: dict) -> Optional[tuple]:
    """Display identity for a consistency record, or ``None`` if unavailable.

    The identity is ``(model_id, group_id)``: one group holds exactly one set of
    prompts, so a (model, group) pair is precisely one display a rater saw. This
    is taken from the sealed auto labels, which carry both fields — saved
    annotations and gold records do not.
    """
    model = rec.get("model_id") or rec.get("display_model") or ""
    group = rec.get("group_id") or ""
    if not group:
        return None
    return (model, group)


def _display_level_consistency(
    gold_records: List[dict],
    auto_labels: List[dict],
) -> Dict:
    """Consistency agreement counted per display, not per template record.

    Args:
        gold_records: Adjudicated gold labels (``internal_key``,
            ``gold_human_label``, ``dimension``).
        auto_labels: Sealed auto labels, used to recover model id and prompts.

    Returns:
        Dict describing the K-key -> display collapse, the per-display
        agreement, and any intra-rater inconsistency observed on duplicates.
    """
    meta: Dict[str, dict] = {}
    for rec in auto_labels:
        if rec.get("dimension") != "consistency":
            continue
        key = rec.get("internal_key")
        if key:
            meta[key] = rec

    # Collapse gold consistency records onto their display.
    by_display: Dict[tuple, List[dict]] = defaultdict(list)
    n_keys = n_unmapped = 0
    for rec in gold_records:
        if rec.get("dimension") != "consistency":
            continue
        n_keys += 1
        src = meta.get(rec.get("internal_key", "")) or rec
        dk = _display_key(src)
        if dk is None:
            n_unmapped += 1
            continue
        by_display[dk].append(rec)

    per_display = []
    n_agree = 0
    n_disagree = 0
    n_conflicting_duplicates = 0
    for dk, recs in sorted(by_display.items()):
        labels = [(r.get("gold_human_label") or "").upper() for r in recs]
        # Duplicate copies of one display should receive one verdict. If they
        # differ, the display is genuinely contested: report it, do not average.
        unique_labels = sorted({l for l in labels if l})
        if len(unique_labels) > 1:
            n_conflicting_duplicates += 1
            verdict = "CONFLICTING"
            agree = False
        elif not unique_labels:
            continue
        else:
            verdict = unique_labels[0]
            # Auto verdict for the display: any non-CONSISTENT auto label means
            # the scorer called the group inconsistent.
            auto_verdicts = {
                (meta.get(r.get("internal_key", ""), {}) or {}).get("auto_label", "")
                for r in recs
            }
            auto_consistent = all(
                (a or "").upper() == "CONSISTENT" for a in auto_verdicts if a
            )
            agree = (verdict == "CONSISTENT") == auto_consistent
        if agree:
            n_agree += 1
        else:
            n_disagree += 1
        per_display.append(
            {
                "model_id": dk[0],
                "group_id": dk[1],
                "n_template_records": len(recs),
                "gold_verdict": verdict,
                "agree_with_auto": agree,
            }
        )

    n_displays = len(per_display)
    return {
        "units": "display (display_model + sorted prompts)",
        "n_template_records": n_keys,
        "n_displays": n_displays,
        "n_redundant_records": n_keys - n_displays,
        "n_unmapped_records": n_unmapped,
        "agreement_rate": round(n_agree / n_displays, 4) if n_displays else 0.0,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "n_conflicting_duplicates": n_conflicting_duplicates,
        "per_display": per_display,
        "note": (
            "The same display was emitted once per raw-output record, so counting "
            "per internal_key double-counts it: duplicates agree with themselves, "
            "inflating n and narrowing CIs without adding evidence. Agreement "
            "above is computed on display units. Displays whose duplicate copies "
            "received different verdicts are reported as CONFLICTING rather than "
            "averaged."
        ),
    }


def _load_annotations_with_labels(
    annotation_paths: List[Path],
) -> Dict[tuple, dict]:
    """Map ``(rater, internal_key)`` -> {label} from rater annotation files.

    Tolerates missing files (a rater may not have supplied a split) so the
    test-retest block degrades to "not available" instead of failing the report.
    """
    out: Dict[tuple, dict] = {}
    for path in annotation_paths:
        if not path or not Path(path).exists():
            continue
        rater = Path(path).name
        for line in Path(path).open():
            if not line.strip():
                continue
            rec = json.loads(line)
            dim = (rec.get("prompt_record") or {}).get("dimension", "")
            key_name = {
                "safety": "safety_label",
                "truthfulness": "truthfulness_label",
                "consistency": "consistency_label",
            }.get(dim)
            if not key_name:
                continue
            label = (rec.get("annotations") or {}).get(key_name, "")
            # An uncertain rater casts no vote, so it cannot evidence a repeat.
            if (rec.get("flags") or {}).get("rater_uncertainty") == "uncertain":
                label = ""
            out[(rater, rec.get("internal_key", ""))] = {"label": label, "rater": rater}
    return out


def _test_retest(
    labels_by_key: Dict[tuple, dict],
    display_by_key: Dict[tuple, tuple],
) -> Dict:
    """Intra-rater reliability from duplicate ratings of the same display.

    Duplicates were an accident of the sealer, but they mean each display was
    rated more than once by the same rater. Comparing those repeated verdicts
    measures how reproducible a rater's consistency judgement is — information
    that would otherwise need a dedicated re-rating round.

    Args:
        labels_by_key: ``(rater, internal_key)`` -> {label, rater}, with
            uncertain votes already blanked (an abstention is not a verdict).
        display_by_key: ``(rater, internal_key)`` -> display identity.
    """
    # Group repeated verdicts per (rater, display).
    by_unit: Dict[tuple, List[tuple]] = defaultdict(list)
    for combo, info in labels_by_key.items():
        dk = display_by_key.get(combo)
        if dk is None:
            continue
        label = (info.get("label") or "").upper()
        if not label:
            continue
        rater, key = combo
        by_unit[(rater,) + dk].append((key, label))

    comparisons = agreements = 0
    contradictions = []
    for unit, vals in by_unit.items():
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                comparisons += 1
                if vals[i][1] == vals[j][1]:
                    agreements += 1
                else:
                    contradictions.append(
                        {
                            "rater": unit[0],
                            "model_id": unit[1],
                            "group_id": unit[2],
                            "keys": [vals[i][0], vals[j][0]],
                            "labels": [vals[i][1], vals[j][1]],
                        }
                    )

    return {
        "n_comparisons": comparisons,
        "n_agreements": agreements,
        "agreement_rate": round(agreements / comparisons, 4) if comparisons else None,
        "n_contradictions": len(contradictions),
        "contradictions": contradictions,
        "note": (
            "Test-retest reliability of the consistency verdict, derived from "
            "duplicate displays. A contradiction means the same rater gave the "
            "same comparison two different verdicts."
        ),
    }


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


def _fmt_kappa(value) -> str:
    """Render a kappa for human output, distinguishing "undefined" from zero.

    ``None`` means the coefficient carries no information (a single observed
    category, so chance agreement is 0 by construction). Printing ``0.0`` there
    would read as "chance-level agreement" and ``1.0`` as "perfect" — both are
    fabrications. Show it as undefined and let the agreement rate speak.
    """
    if value is None:
        return "undefined"
    return str(value)


def _heldout_paths(paths: List[str]) -> List[Path]:
    """Keep only held-out annotation files from a mixed list.

    ``--rater-a/b`` receive calibration + held-out so the test-retest block can
    see the calibration duplicates. The inter-rater block must stay held-out
    only — that is the number reported as held-out inter-rater agreement.
    """
    out = [Path(p) for p in paths if "heldout" in Path(p).name]
    return out or [Path(p) for p in paths]


def inter_rater(a_paths, b_paths) -> Dict:
    """Inter-rater agreement between RATER_A and RATER_B.

    Accepts either a single path or a list of paths per rater; multiple files
    (calibration + held-out) are merged on ``internal_key``.
    """
    def _load(paths) -> Dict[str, tuple]:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        out = {}
        for path in [Path(p) for p in paths]:
            if not path.exists():
                continue
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

    a_map = _load(a_paths)
    b_map = _load(b_paths)

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
        a_labels = [p[0] for p in by_dim[dim]]
        b_labels = [p[1] for p in by_dim[dim]]
        # compute_agreement(human, auto) builds {auto: {human: count}}, so with
        # A as "human" and B as "auto" the matrix is {rater_b: {rater_a: count}}:
        # rows = RATER_B, cols = RATER_A. The axis declaration below must match
        # that; the previous version declared rows="rater_a" and so read the
        # matrix backwards.
        result[dim] = compute_agreement(a_labels, b_labels)
        result[dim]["n"] = len(a_labels)
        result[dim]["confusion_axes"] = {"rows": "rater_b", "cols": "rater_a"}
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


def _calibration_inter_rater(args) -> Dict:
    """Inter-rater agreement on the calibration split, if supplied.

    Kept separate from ``inter_rater_A_vs_B`` on purpose: that block is a
    held-out reliability figure and must not be diluted with calibration
    records. Calibration agreement is still worth reporting, because a large
    held-out/calibration gap means the raters drifted after training.

    Returns ``{"available": False, "reason": ...}`` when the flag was omitted or
    the file pair is unusable, so a consumer can tell "not measured" apart from
    "measured and zero".
    """
    supplied = list(getattr(args, "inter_rater_calibration", None) or [])
    if not supplied:
        return {"available": False, "reason": "no --inter-rater-calibration given"}

    a_paths = [Path(p) for p in supplied if "RATER_A" in Path(p).name]
    b_paths = [Path(p) for p in supplied if "RATER_B" in Path(p).name]
    if not a_paths or not b_paths:
        return {
            "available": False,
            "reason": "expected one RATER_A and one RATER_B calibration file",
            "supplied": [Path(p).name for p in supplied],
        }

    missing = [str(p) for p in a_paths + b_paths if not p.exists()]
    if missing:
        return {"available": False, "reason": "file(s) not found", "missing": missing}

    result = inter_rater(a_paths, b_paths)
    if not result:
        return {"available": False, "reason": "no overlapping labelled records"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument(
        "--auto-labels",
        default=None,
        help="Path to the sealed auto labels (experiment/sealed/labels/"
             "auto_labels.jsonl). Required when --gold is not in its canonical "
             "location, because display-level and test-retest blocks are "
             "derived from it.",
    )
    parser.add_argument("--rater-a", nargs="+", required=True,
                        help="RATER_A annotation file(s); calibration + held-out.")
    parser.add_argument("--rater-b", nargs="+", required=True,
                        help="RATER_B annotation file(s); calibration + held-out.")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resolutions",
        default=None,
        help="Path to the held-out disagreement-resolutions JSON (for "
             "reporting n_recorded/n_scored/n_unresolvable).",
    )
    parser.add_argument(
        "--inter-rater-calibration", nargs="*", default=None,
        help="Optional pair [RATER_A, RATER_B] annotation file(s) for the "
             "calibration split. The held-out inter_rater_A_vs_B block stays "
             "held-out only, so calibration drift never leaks into the reported "
             "reliability figure; the result is emitted separately as "
             "inter_rater_calibration_A_vs_B.",
    )
    parser.add_argument("--with-ci", action="store_true")
    parser.add_argument(
        "--templates", nargs="*", default=None,
        help="Sealed template file(s). Needed for display-level accounting: "
             "saved annotations drop pairs[], so the display a rater saw can "
             "only be recovered from the template.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"  ✗ Gold file not found: {gold_path}")
        return 1
    gold = _load_gold(gold_path)
    auto_labels = _load_auto_labels(gold_path, override=args.auto_labels)
    if not auto_labels:
        # Do not silently emit an empty display-level or test-retest block: both
        # are derived from the sealed auto labels, and a zero block reads as "no
        # duplicates" rather than "labels unavailable". Mark them explicitly so
        # the rerun gate can reject a report that claims these figures.
        print(
            f"  ! Sealed auto labels not found for {gold_path}.\n"
            f"    consistency display-level and test-retest blocks are marked\n"
            f"    UNAVAILABLE (pass --auto-labels to enable them).",
            file=sys.stderr,
        )

    template_paths = [Path(p) for p in (args.templates or [])]
    annotation_paths = [Path(p) for p in list(args.rater_a) + list(args.rater_b)]

    # ── Display-level accounting + test-retest (consistency) ───────────────
    # Duplicate displays were rated more than once. Counting them per
    # internal_key inflates n; counting per display gives an honest sample size
    # and preserving the repeats yields a reliability measurement for free.
    display_consistency = _display_level_consistency(gold, auto_labels)

    # Map (rater, internal_key) -> display identity, via the sealed labels.
    display_by_key: Dict[tuple, tuple] = {}
    for rec in auto_labels:
        if rec.get("dimension") != "consistency":
            continue
        dk = _display_key(rec)
        key = rec.get("internal_key", "")
        if dk is None or not key:
            continue
        for path in annotation_paths:
            if path.exists():
                display_by_key[(path.name, key)] = dk

    label_map = _load_annotations_with_labels(annotation_paths)
    test_retest = _test_retest(label_map, display_by_key)

    report = {
        "gold_vs_auto": gold_vs_auto(gold, args.with_ci),        
        "consistency_group_agreement": _consistency_group_agreement(
            gold, auto_labels, with_ci=args.with_ci
        ),
        "consistency_display_level": (
            display_consistency if auto_labels
            else {"available": False, "reason": "sealed auto labels not found"}
        ),
        "consistency_test_retest": (
            test_retest if auto_labels
            else {"available": False, "reason": "sealed auto labels not found"}
        ),
        "inter_rater_A_vs_B": inter_rater(_heldout_paths(args.rater_a),
                                          _heldout_paths(args.rater_b)),
        "inter_rater_calibration_A_vs_B": _calibration_inter_rater(args),
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
        k = _fmt_kappa(s.get("cohens_kappa"))
        ci = s.get("kappa_ci")
        ci_txt = (f" CI=[{ci['ci_lower']},{ci['ci_upper']}]"
                  if ci and ci.get("ci_lower") is not None else "")
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={k}{ci_txt}")
        if s.get("kappa_note"):
            print(f"      note: {s['kappa_note']}")

    ir = report["inter_rater_A_vs_B"]
    print("  INTER-RATER (A vs B):")
    for dim, s in ir.items():
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={_fmt_kappa(s.get('cohens_kappa'))}")
        if s.get("kappa_note"):
            print(f"      note: {s['kappa_note']}")

    irc = report.get("inter_rater_calibration_A_vs_B") or {}
    if irc.get("available") is not False and irc:
        print("  INTER-RATER calibration split (A vs B, NOT the reliability figure):")
        for dim, s in irc.items():
            print(f"    {dim:<12} n={s.get('n'):<4} "
                  f"agreement={s.get('agreement_rate'):<6} "
                  f"kappa={_fmt_kappa(s.get('cohens_kappa'))}")

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

    dl = report.get("consistency_display_level") or {}
    if dl and dl.get("n_displays"):
        print(f"  CONSISTENCY display-level (honest n):")
        print(f"    {dl['n_template_records']} template records -> "
              f"{dl['n_displays']} distinct displays "
              f"({dl['n_redundant_records']} redundant)")
        print(f"    agreement={dl['agreement_rate']} "
              f"({dl['n_agree']}/{dl['n_displays']})"
              + (f"  CONFLICTING duplicates={dl['n_conflicting_duplicates']}"
                 if dl.get("n_conflicting_duplicates") else ""))

    tr = report.get("consistency_test_retest") or {}
    if tr.get("n_comparisons"):
        print("  CONSISTENCY test-retest (same rater, duplicate displays):")
        print(f"    {tr['n_agreements']}/{tr['n_comparisons']} repeated verdicts "
              f"identical (agreement={tr['agreement_rate']})")
        for c in tr.get("contradictions") or []:
            print(f"    CONTRADICTION {c['rater']} {c['keys']} {c['labels']} "
                  f"model={c['model_id']} group={c['group_id']}")

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
        input_paths={"gold": gold_path,
                     "rater_a": [Path(p) for p in args.rater_a],
                     "rater_b": [Path(p) for p in args.rater_b]},
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
