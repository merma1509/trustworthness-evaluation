#!/usr/bin/env python3
"""Budget Optimizer — turn auto-scorer reliability (κ) into a human-budget plan.

Implements the "Trust Budget" algorithm:
instead of spending human annotation uniformly across dimensions, allocate it
where the auto-scorer is least reliable.

Input
-----
Per-dimension auto–human Cohen's κ, read from either:

* the **blinded experiment report** produced by
  ``scripts/run_blinded_annotation.py report``
  (``experiment/held_out_agreement_report.json`` — preferred, anonymised flow), or
* the **validation report** (``results/validation_report.json``, RQ1
  ``by_dimension`` — the calibration estimate).

The dimension-level κ drives the recommendation; a global κ (overall auto-vs-gold)
is used when a specific dimension's estimate is missing.

Output
------
A budget plan (``budget_plan.json`` by default) mapping each dimension to a
``TrustBand`` and the number/estimated cost of human annotations to route there.

Usage
-----
    # Use the blinded experiment report (after it exists):
    python3 scripts/budget_optimizer.py \
        --report experiment/held_out_agreement_report.json \
        --output results/budget_plan.json

    # Use the calibration validation report:
    python3 scripts/budget_optimizer.py \
        --report results/validation_report.json

    # Override the κ gates:
    python3 scripts/budget_optimizer.py --gate-trust 0.7 --gate-unverified 0.4
"""

import argparse
import json
from pathlib import Path

# Three-band "trust region". Gates are deliberately configurable
# so a sensitivity note can be written in the paper.
DEFAULT_GATE_TRUST = 0.7  # κ >= this -> auto-scorer used directly
DEFAULT_GATE_UNVERIFIED = 0.4  # κ <  this -> dimension routed to human annotation
DEFAULT_GATES = {
    "trust": DEFAULT_GATE_TRUST,
    "unverified": DEFAULT_GATE_UNVERIFIED,
}

# Holistic budget maths (rq4_cost): human $20/h.
HUMAN_HOURLY_COST = 20.0

# Fallback placeholder if no timing study file is present (same default the
# validation report uses). A genuine study overrides this when available.
DEFAULT_HUMAN_SECONDS_PER_LABEL = 30.0
HUMAN_TIMING_PATH = Path("results/human_timing_measurement.json")


def _load_human_seconds_per_label() -> float:
    """Load the human annotation timing from ``human_timing_measurement.json``.

    The budget/cost analysis must share a *single source of truth* for the
    measured human time per label. If ``results/human_timing_measurement.json``
    (written by ``scripts/measure_human_annotation_time.py``, Task 1.5) exists,
    its ``median_seconds_per_label`` is used; otherwise we fall back to the
    same 30 s placeholder used by the validation report, so the budget and the
    cost ratio are mutually consistent and never silently diverge.
    """
    if HUMAN_TIMING_PATH.exists():
        try:
            with HUMAN_TIMING_PATH.open() as f:
                data = json.load(f)
            sec = data.get("median_seconds_per_label")
            if sec:
                return float(sec)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_HUMAN_SECONDS_PER_LABEL


# Resolve once at import time so every ``build_plan`` call uses the same value
# (matching what the validation report would have used).
HUMAN_SECONDS_PER_LABEL = _load_human_seconds_per_label()


def _load_report(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _extract_dimension_kappas(report: dict, source: str) -> dict:
    """Return ``{dimension: kappa}`` from a report, tolerant of two formats.

    Supports:
      * experiment (``run_blinded_annotation``): ``by_dimension[dim].auto_comparison.cohens_kappa``
      * validation (``paradigm_report``): ``rq1_agreement.by_dimension[dim].cohens_kappa``
    """
    out: dict = {}

    # Blinded experiment report: auto-comparison is the gold-vs-auto headline κ.
    bd = report.get("by_dimension")
    if isinstance(bd, dict) and bd:
        for dim, sub in bd.items():
            auto_cmp = (sub or {}).get("auto_comparison") or {}
            k = auto_cmp.get("cohens_kappa")
            if k is not None:
                out[dim] = k
            else:
                # Fall back to the inter-annotator gate when auto-comparison absent.
                ia = (sub or {}).get("inter_annotator") or {}
                if ia.get("mean_kappa") is not None:
                    out[dim] = ia["mean_kappa"]
        if out:
            return out

    # Validation report (RQ1 by_dimension).
    rq1 = report.get("rq1_agreement") or {}
    bd2 = rq1.get("by_dimension")
    if isinstance(bd2, dict):
        for dim, sub in bd2.items():
            k = (sub or {}).get("cohens_kappa")
            if k is not None:
                out[dim] = k
        if out:
            return out

    # Overall fallback — single global κ applied to all dimensions.
    overall = report.get("overall") or rq1.get("overall") or {}
    auto_cmp = overall.get("auto_comparison") or {}
    k_global = auto_cmp.get("cohens_kappa")
    if k_global is None and "cohens_kappa" in overall:
        k_global = overall["cohens_kappa"]
    if k_global is not None:
        for dim in ("safety", "truthfulness", "consistency"):
            out[dim] = k_global
    return out


def _band(kappa: float, gates: dict) -> str:
    """Map a κ value to one of the three trust bands."""
    if kappa >= gates["trust"]:
        return "trust"
    if kappa >= gates["unverified"]:
        return "caveated"
    return "unverified"


def _records_per_dimension(report: dict, dim: str) -> int:
    """Best-effort count of annotatable records for a dimension."""
    # Experiment report: adjudicated n for the dimension, else overall.
    bd = report.get("by_dimension") or {}
    sub = bd.get(dim) or {}
    adj = (sub or {}).get("adjudicated") or {}
    n = adj.get("n_total")
    if n:
        return int(n)
    # Validation report RQ1 overall n.
    rq1 = report.get("rq1_agreement") or {}
    bd2 = rq1.get("by_dimension") or {}
    n2 = (bd2.get(dim) or {}).get("n")
    if n2:
        return int(n2)
    return 0


def build_plan(report: dict, gates: dict, source: str) -> dict:
    """Compute the budget plan from a report dict.

    Args:
        report: Loaded JSON report (experiment or validation).
        gates: ``{"trust": float, "unverified": float}``.
        source: Label describing the report source (for the ``note`` field).

    Returns:
        A plan dict with per-dimension bands, annotations needed, and cost.
    """
    kappas = _extract_dimension_kappas(report, source)
    plan_records = []
    total_labels = 0
    for dim in ("safety", "truthfulness", "consistency"):
        k = kappas.get(dim)
        n = _records_per_dimension(report, dim)
        if k is None:
            plan_records.append(
                {
                    "dimension": dim,
                    "kappa": None,
                    "band": "unknown",
                    "recommendation": "No κ estimate available — default to sampling.",
                    "annotations_needed": n if n else None,
                }
            )
            continue
        band = _band(k, gates)
        if band == "unverified":
            # Route ALL records in this dimension to human annotation.
            needed = n if n else None
            recommendation = (
                "Auto-scorer unreliable here (κ < gate). Route full dimension to human annotation."
            )
        elif band == "caveated":
            # Sample a fixed 10% (spot-check) rather than full review.
            needed = round(n * 0.10) if n else None
            recommendation = (
                "Auto-scorer usable but caveated. Spot-check ~10% of records; "
                "widen CIs and re-check periodically."
            )
        else:
            needed = 0
            recommendation = (
                "Auto-scorer reliable (κ >= trust gate). No human budget needed; "
                "optional small drift-check sample."
            )
        if needed:
            total_labels += needed
        plan_records.append(
            {
                "dimension": dim,
                "kappa": round(k, 4),
                "band": band,
                "annotations_needed": needed,
                "recommendation": recommendation,
            }
        )

    cost = round(total_labels * (HUMAN_SECONDS_PER_LABEL / 3600) * HUMAN_HOURLY_COST, 2)
    return {
        "source": source,
        "gates": gates,
        "human_seconds_per_label": HUMAN_SECONDS_PER_LABEL,
        "human_timing_source": (
            "results/human_timing_measurement.json"
            if HUMAN_TIMING_PATH.exists() and HUMAN_TIMING_PATH.stat().st_size > 0
            else "default placeholder (30.0 s/label) — no timing study file present"
        ),
        "by_dimension": plan_records,
        "total_human_annotations": total_labels,
        "estimated_human_cost_usd": cost,
        "note": (
            "Trust Budget policy (Part 3 §4.2/§8.1): allocate human annotation only "
            "where the auto-scorer is below the trust κ gate. 'unverified' → full "
            "dimension to humans; 'caveated' → ~10% spot-check; 'trust' → none. "
            "Human cost is derived from human_seconds_per_label above, which reads "
            "results/human_timing_measurement.json when present (Task 1.5), else "
            "falls back to the 30 s placeholder to stay consistent with rq4_cost."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", required=True, help="Path to experiment or validation report JSON."
    )
    parser.add_argument(
        "--output", default="results/budget_plan.json", help="Where to write the plan."
    )
    parser.add_argument(
        "--gate-trust",
        type=float,
        default=DEFAULT_GATE_TRUST,
        help=f"κ at/above this trusts the auto-scorer (default {DEFAULT_GATE_TRUST}).",
    )
    parser.add_argument(
        "--gate-unverified",
        type=float,
        default=DEFAULT_GATE_UNVERIFIED,
        help=f"κ below this routes the dimension to humans (default {DEFAULT_GATE_UNVERIFIED}).",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"✗ Report not found: {report_path}")
        return 1

    gates = {"trust": args.gate_trust, "unverified": args.gate_unverified}
    report = _load_report(report_path)
    plan = build_plan(report, gates, source=str(report_path))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"\nBudget plan written to {out}")
    print(f"Gates: trust κ ≥ {args.gate_trust}, unverified κ < {args.gate_unverified}\n")
    for rec in plan["by_dimension"]:
        k = f"{rec['kappa']:.3f}" if rec["kappa"] is not None else "n/a"
        print(
            f"  {rec['dimension']:<14} κ={k:<7} band={rec['band']:<10} "
            f"annotations={rec['annotations_needed']}"
        )
    print(f"\n  TOTAL human annotations: {plan['total_human_annotations']}")
    print(f"  Estimated cost: ${plan['estimated_human_cost_usd']:.2f} @ $20/h")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
