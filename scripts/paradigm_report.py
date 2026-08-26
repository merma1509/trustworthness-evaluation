#!/usr/bin/env python3
"""Paradigm Report — Measurement Validation for Trustworthiness Evaluation

Generates a comprehensive report addressing 4 research questions:

    RQ1: How well do auto-labels agree with human labels?
    RQ2: What factors affect agreement? (dimension, attack type, model)
    RQ3: What dataset size is needed for stable scores?
    RQ4: What is the cost of automatic vs. human evaluation?

Usage:
    python3 scripts/paradigm_report.py \
        --audit results/audit/all_audit.jsonl \
        --output results/validation_report.json

    # With cost tracker data
    python3 scripts/paradigm_report.py --with-cost

Output:
    results/validation_report.json   (machine-readable)
    Console: formatted report
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DIMENSIONS, RAW_OUTPUTS_DIR, RESULTS_DIR
from src.utils import load_jsonl
from src.validation import compute_validation_report

# ──────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────


def _extract_per_prompt_scores() -> dict:
    """Extract per-UNIT 0/1 scores for each dimension (RQ3 unit of analysis).

    **Independent observations, NOT raw records:**

    - ``safety`` / ``truthfulness``: one observation per **unique prompt_id**
      (each prompt is answered once per model, so a prompt_id never repeats
      within a dimension file).
    - ``consistency``: one observation per **multi-prompt group** — the group
      is the cluster / independent unit, and the records *within* a group are
      dependent (they share a prompt family). Singleton (benign) groups are
      excluded because they are never scored.

    This fixes the earlier flaw where raw records were double-counted across
    the two models (consistency appeared as 64 records instead of 22
    independent groups) and where repeated prompt texts inflated the unit
    count.

    Returns ``{dim: [0/1, ...]}`` of independent unit scores.
    """
    dims = {dim: [] for dim in DIMENSIONS}

    for dim in DIMENSIONS:
        seen = set()
        for model_file in RAW_OUTPUTS_DIR.glob(f"*_{dim}.jsonl"):
            with open(model_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if dim == "consistency":
                        # Unit = group. Skip singletons (never scored).
                        gid = r.get("group_id")
                        if gid is None or r.get("is_singleton"):
                            continue
                        if gid in seen:
                            continue
                        seen.add(gid)
                        dims[dim].append(1.0 if r.get("group_consistent") else 0.0)
                    else:
                        # Unit = unique prompt.
                        pid = r.get("prompt_id") or r.get("prompt_text")
                        if pid is None or pid in seen:
                            continue
                        seen.add(pid)
                        dims[dim].append(1.0 if r.get("is_correct", False) else 0.0)
    return dims


def _load_cost_data() -> dict:
    """Try to load cost tracker data if available."""
    cost_path = RESULTS_DIR / "cost_tracker.json"
    if cost_path.exists():
        with open(cost_path) as f:
            return json.load(f)
    return None


def _load_human_timing() -> dict:
    """Try to load a human-annotation timing study (Task 1.5).

    Written by ``scripts/measure_human_annotation_time.py`` (live interactive
    timing; the only way this file is produced). When present it replaces the
    30 s placeholder for the cost ratio. Whether that value counts as genuinely
    "MEASURED" is decided by its ``measurement_validity`` field (== "MEASURED").
    """
    path = RESULTS_DIR / "human_timing_measurement.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return None


def _format_agreement_badge(kappa: float) -> str:
    """Return human-readable interpretation of Cohen's Kappa."""
    if kappa >= 0.81:
        return "Almost perfect"
    elif kappa >= 0.61:
        return "Substantial"
    elif kappa >= 0.41:
        return "Moderate"
    elif kappa >= 0.21:
        return "Fair"
    elif kappa >= 0.0:
        return "Slight"
    else:
        return "Poor (worse than random)"


def print_report(report: dict):
    """Print formatted report to console."""
    print()
    print("=" * 72)
    print("  MEASUREMENT VALIDATION REPORT")
    print('  Paradigm shift: "When can we trust a small local evaluation?"')
    print("=" * 72)

    meta = report.get("meta", {})
    print(
        f"\n  Meta: {meta.get('labelled_records', 0)}/{meta.get('total_audit_records', 0)} "
        f"records labelled ({meta.get('pct_labelled', 0)}%)"
    )

    # ── RQ1 ──────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  RQ1: How well do the automatic labels match the human ones?")
    print(f"{'─' * 72}")

    rq1 = report.get("rq1_agreement", {})
    if rq1 and "overall" in rq1:
        oa = rq1["overall"]
        print("\n  Overall:")
        print(f"    Agreement rate:  {oa.get('agreement_rate', 0) * 100:.1f}%")
        print(
            f"    Cohen's Kappa:   {oa.get('cohens_kappa', 0):.4f}  ({_format_agreement_badge(oa.get('cohens_kappa', 0))})"
        )
        print(f"    n = {oa.get('n_valid_pairs', 0)}")

        if "by_dimension" in rq1:
            print("\n  Per dimension:")
            for dim, da in sorted(rq1["by_dimension"].items()):
                badge = _format_agreement_badge(da.get("cohens_kappa", 0))
                print(f"    {dim:<15}  n={da['n']:<3}  κ={da['cohens_kappa']:.3f}  ({badge})")

        if "by_attack_type" in rq1:
            print("\n  Per attack type:")
            for atype, da in sorted(rq1["by_attack_type"].items()):
                badge = _format_agreement_badge(da.get("cohens_kappa", 0))
                print(f"    {atype:<25}  n={da['n']:<3}  κ={da['cohens_kappa']:.3f}  ({badge})")

        fp = rq1.get("false_positives", [])
        fn = rq1.get("false_negatives", [])
        print("\n  Error analysis:")
        print(f"    False positives (auto too optimistic): {len(fp)}")
        print(f"    False negatives (auto too pessimistic): {len(fn)}")
    else:
        print("\n No labelled data. Run annotation first.")

    # ── RQ2 ──────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  RQ2: Which factors influence agreement?")
    print(f"{'─' * 72}")

    rq2 = report.get("rq2_factors", {})
    if rq2:
        # Find worst agreement
        all_factors = []
        for factor_name, subgroups in rq2.items():
            for subname, stats in subgroups.items():
                all_factors.append((factor_name, subname, stats["cohens_kappa"], stats["n"]))
        all_factors.sort(key=lambda x: x[2])  # worst first

        if all_factors:
            worst = all_factors[0]
            best = all_factors[-1]
            print(f"\n  Worst agreement:  {worst[0]}={worst[1]}  (κ={worst[2]:.3f}, n={worst[3]})")
            print(f"  Best agreement:   {best[0]}={best[1]}  (κ={best[2]:.3f}, n={best[3]})")

            print("\n  All factors (sorted by κ):")
            for factor, sub, k, n in all_factors:
                bar = "█" * int(abs(k) * 30)
                print(f"    {factor:<15} {sub:<25} κ={k:.3f}  n={n:<3}  {bar}")
    else:
        print("\n No factor data available.")

    # ── RQ3 ──────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  RQ3: What minimum dataset size is needed for a stable evaluation?")
    print(f"{'─' * 72}")

    rq3 = report.get("rq3_dataset_stability", {})
    if rq3:
        for dim, ds in rq3.items():
            jackknife = ds.get("jackknife_stability", {})
            size_sens = ds.get("dataset_size_sensitivity", {})

            print(f"\n  {dim}:")
            print(
                f"    Full score: {jackknife.get('full_score', 0):.4f} (n={jackknife.get('n_total', 0)})"
            )

            if jackknife:
                std = jackknife.get("std_jackknife", 0)
                print(f"    Jackknife std (leave-1-out): ±{std:.4f}")
                print(f"    Max decrease from full score: {jackknife.get('max_decrease', 0):.4f}")

            if size_sens:
                print(
                    f"    Estimated min N for CI width < 0.10: {size_sens.get('estimated_min_n', '?')}"
                )
                print(f"    Current CI width at n={size_sens.get('full_n', '?')}: ", end="")
                sizes = size_sens.get("sizes", [])
                if sizes:
                    final = sizes[-1]
                    print(f"{final.get('ci_width', '?'):.4f}")
                else:
                    print("N/A")

            # Defensible future-N (independent units; group-level for consistency)
            req10 = ds.get("required_n_ci_width_0_10", {})
            req05 = ds.get("required_n_ci_width_0_05", {})
            if req10:
                print(f"    Unit of analysis: {ds.get('unit_of_analysis', '?')}")
                print(f"    Required N for +/-10% CI @ 95%: {req10.get('n_required', '?')}")
                print(f"    Required N for +/-5%  CI @ 95%: {req05.get('n_required', '?')}")
                req_emp = ds.get("required_n_ci_width_0_05_empirical", {})
                print(
                    f"    Required N for +/-5%  CI @ 95% (EMPIRICAL, bootstrap): "
                    f"{req_emp.get('n_required', '?')}"
                )

            # Summary finding
            if jackknife:
                std = jackknife.get("std_jackknife", 0)
                if std < 0.03:
                    print("    → STABLE: removing 1 prompt changes score by < 3%")
                elif std < 0.05:
                    print("    → MODERATE: removing 1 prompt changes score by < 5%")
                else:
                    print("    → UNSTABLE: score is sensitive to individual prompts")
    else:
        print("\n No stability data available.")

    # ── RQ4 ──────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  RQ4: How much does automatic evaluation cost vs. manual?")
    print(f"{'─' * 72}")

    rq4 = report.get("rq4_cost", {})
    if rq4:
        print(
            f"\n  Parameters: {rq4.get('parameters', {}).get('total_responses', 0)} total responses "
            f"({rq4.get('parameters', {}).get('num_prompts', 0)} prompts × "
            f"{rq4.get('parameters', {}).get('num_models', 0)} models)"
        )

        auto = rq4.get("fully_automatic", {})
        human = rq4.get("fully_human", {})
        hybrid = rq4.get("hybrid_50pct_audit", {})

        h1 = "Time"
        h2 = "Cost"
        h3 = "Notes"
        print(f"\n    {'':<25} {h1:>8} {h2:>8}  {h3:<20}")
        sep = "-" * 65
        print(f"    {sep}")

        label_a = "Fully automatic"
        label_h = "Fully human"
        label_hyb = "Hybrid (50% audit)"
        auto_note = "All models x all prompts"
        human_note = "All labels by annotator"
        hybrid_note = "Auto 100% + human 50%"
        auto_t = auto.get("time_hours", 0)
        auto_c = auto.get("cost", 0)
        human_t = human.get("time_hours", 0)
        human_c = human.get("cost", 0)
        hybrid_t = hybrid.get("time_hours", 0)
        hybrid_c = hybrid.get("cost", 0)
        print(f"    {label_a:<25} {auto_t:>6.1f}h  ${auto_c:>6.2f}  {auto_note}")
        print(f"    {label_h:<25} {human_t:>6.1f}h  ${human_c:>6.2f}  {human_note}")
        print(f"    {label_hyb:<25} {hybrid_t:>6.1f}h  ${hybrid_c:>6.2f}  {hybrid_note}")

        params = rq4.get("parameters", {})
        meas_flag = "MEASURED" if rq4.get("cost_is_measured") else "ESTIMATED"
        print(
            f"\n    Human time per label: {rq4.get('parameters', {}).get('human_time_per_label_sec', '?')}s  "
            f"[{params.get('human_time_source', 'default placeholder')}]"
        )
        print(f"    Cost basis: {meas_flag}")

        print("\n  Recommendation:")
        print(f"    {rq4.get('recommendation', '')}")

        measured = rq4.get("measured_times")
        if measured:
            print("\n  Measured times:")
            for rec in measured.get("records", []):
                print(
                    f"    {rec.get('label', '')}: {rec.get('elapsed_seconds', 0)}s "
                    f"({rec.get('seconds_per_prompt', 0):.2f}s/prompt)"
                )
    else:
        print("\n No cost data available.")

    print(f"\n{'=' * 72}")
    print("  END OF VALIDATION REPORT")
    print(f"{'=' * 72}\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Measurement Validation Report (Paradigm Shift)")
    parser.add_argument(
        "--audit",
        "-a",
        type=str,
        default="results/audit/all_audit.jsonl",
        help="Path to audit JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/validation_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--with-cost",
        action="store_true",
        help="Include cost tracker data if available",
    )
    parser.add_argument(
        "--auto-time-per-prompt",
        type=float,
        default=None,
        help="MEASURED average auto-inference seconds per prompt. "
        "Overrides the default when not using --with-cost.",
    )
    parser.add_argument(
        "--human-time-per-label",
        type=float,
        default=30.0,
        help="MEASURED average human-annotation seconds per label. "
        "Fill from a real timing study (Task 1.5) for a definitive ratio.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-record information",
    )
    args = parser.parse_args()

    # Load audit records
    if not Path(args.audit).exists():
        print(f"  Audit file not found: {args.audit}")
        print("  Generate first: python3 scripts/generate_audit_samples.py")
        sys.exit(1)

    audit_records = load_jsonl(args.audit)
    print(f"  Loaded {len(audit_records)} audit records from {args.audit}")

    # Load per-prompt scores
    per_prompt_scores = _extract_per_prompt_scores()
    print(
        f"  Per-prompt scores loaded: "
        f"safety={len(per_prompt_scores.get('safety', []))}, "
        f"truthfulness={len(per_prompt_scores.get('truthfulness', []))}, "
        f"consistency={len(per_prompt_scores.get('consistency', []))}"
    )

    # Load cost data
    cost_data = None
    if args.with_cost:
        cost_data = _load_cost_data()
        if cost_data:
            print(f"  Cost tracker data loaded: {cost_data.get('total_prompts', 0)} prompts")

    # MEASURED auto-time: prefer the CostTracker's measured avg seconds/prompt
    # (results/cost_tracker.json produced by a real run), then an explicit CLI
    # override, then the default.
    measured_auto = (
        cost_data.get("avg_seconds_per_prompt")
        if cost_data and cost_data.get("avg_seconds_per_prompt")
        else None
    )
    auto_time = (
        args.auto_time_per_prompt
        if args.auto_time_per_prompt is not None
        else (measured_auto if measured_auto is not None else 10.0)
    )
    auto_src = (
        "MEASURED (--auto-time-per-prompt)"
        if args.auto_time_per_prompt is not None
        else ("MEASURED (cost_tracker.json)" if measured_auto is not None else "default (assumed)")
    )
    print(f"  Auto time per prompt: {auto_time:.2f}s  [{auto_src}]")

    # Load a MEASURED human-annotation timing study (Task 1.5), if one has been
    # produced. When present it overrides the 30 s placeholder for the ratio.
    human_timing = _load_human_timing()
    if human_timing:
        validity = (human_timing.get("measurement_validity") or "").upper()
        label = (
            "MEASURED"
            if validity == "MEASURED"
            else f"{validity or 'EMULATED/PLACEHOLDER'} (not a verified interactive run)"
        )
        print(
            f"  Human timing loaded: {human_timing.get('median_seconds_per_label', '?')}s/label "
            f"(n={human_timing.get('n_measured', '?')}) from "
            f"{RESULTS_DIR / 'human_timing_measurement.json'} "
            f"[{label}]"
        )
    else:
        print(
            "  Human timing: none found — using default 30.0s/label. "
            "Run scripts/measure_human_annotation_time.py for a MEASURED value (Task 1.5)."
        )

    # Effective human time: an explicit CLI override wins, else the measured
    # timing study, else the default placeholder.
    human_time = (
        args.human_time_per_label
        if args.human_time_per_label != 30.0
        else (
            human_timing.get("median_seconds_per_label")
            if human_timing and human_timing.get("median_seconds_per_label")
            else 30.0
        )
    )
    human_src = (
        "MEASURED (--human-time-per-label)"
        if args.human_time_per_label != 30.0
        else (
            (
                f"MEASURED (human_timing_measurement.json: {human_timing.get('measured_at', '?')})"
                if (human_timing.get("measurement_validity") or "").upper() == "MEASURED"
                else (
                    f"{(human_timing.get('measurement_validity') or 'EMULATED/PLACEHOLDER').upper()} "
                    f"(human_timing_measurement.json: {human_timing.get('measured_at', '?')}) — "
                    f"not a verified interactive timing study"
                )
            )
            if human_timing and human_timing.get("median_seconds_per_label")
            else "default placeholder (not yet measured)"
        )
    )
    print(f"  Human time per label: {human_time:.1f}s  [{human_src}]")

    # Compute validation report
    print("\n  Computing validation report...")
    report = compute_validation_report(
        audit_records=audit_records,
        per_prompt_scores=per_prompt_scores,
        cost_tracker_data=cost_data,
        auto_time_per_prompt=auto_time,
        human_time_per_label=human_time,
        measured_human_timing=human_timing
        if human_timing and human_timing.get("median_seconds_per_label")
        else None,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to {out_path}")

    # ── Also save agreement_report.json (for dashboard) ──
    rq1_data = report.get("rq1_agreement", {})
    if rq1_data and "overall" in rq1_data:
        overall = rq1_data["overall"]
        agreement_out = {
            "pipeline": "paradigm_report.py",
            "total_records": overall.get("n_valid_pairs", overall.get("n", 0)),
            "labelled_records": overall.get("n_valid_pairs", overall.get("n", 0)),
            "overall": overall,
            "by_dimension": rq1_data.get("by_dimension", {}),
            "by_model": rq1_data.get("by_model", {}),
            "by_attack_type": rq1_data.get("by_attack_type", {}),
        }
        agreement_path = out_path.parent / "audit" / "agreement_report.json"
        agreement_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agreement_path, "w") as f:
            json.dump(agreement_out, f, indent=2, ensure_ascii=False)
        n = agreement_out["total_records"]
        k = overall.get("cohens_kappa", 0)
        print(f"  Agreement report saved to {agreement_path} (n={n}, κ={k:.4f})")
    else:
        print("  No agreement data found — skipping agreement_report.json")

    # Print formatted report
    print_report(report)

    # Final summary
    print("\n  Summary:")
    rq1 = report.get("rq1_agreement", {}).get("overall", {})
    rq3 = report.get("rq3_dataset_stability", {})
    if rq3:
        avg_kappa = rq1.get("cohens_kappa", 0)
        print(
            f"  RQ1: Auto-human agreement: k={avg_kappa:.3f} ({_format_agreement_badge(avg_kappa)})"
        )
        # Find min dimensions stability
        stable_dims = [
            d
            for d, ds in rq3.items()
            if ds.get("jackknife_stability", {}).get("std_jackknife", 1) < 0.05
        ]
        unstable_dims = [
            d
            for d, ds in rq3.items()
            if ds.get("jackknife_stability", {}).get("std_jackknife", 1) >= 0.05
        ]
        if stable_dims:
            print(f"  RQ3: Stable dimensions: {', '.join(stable_dims)}")
        if unstable_dims:
            print(f"  RQ3: Unstable dimensions: {', '.join(unstable_dims)} (NEED MORE DATA)")
    human_cost = report.get("rq4_cost", {}).get("fully_human", {}).get("cost", 1)
    auto_cost = max(report.get("rq4_cost", {}).get("fully_automatic", {}).get("cost", 0.01), 0.01)
    ratio = human_cost / auto_cost
    print(f"  RQ4: Cost ratio auto/human = {ratio:.0f}x")
    print()


if __name__ == "__main__":
    main()
