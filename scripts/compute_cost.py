#!/usr/bin/env python3
"""Accurate cost breakdown for auto vs. human vs. hybrid evaluation.

The goal is to stop conflating three different
things — measured facts, working assumptions, and model estimates — into a single
"× cheaper" headline.

Every returned number is explicitly tagged as one of:

* **MEASURED**  — directly observed from the pipeline / timing study.
* **ASSUMED**   — a working assumption with a stated value (e.g. $20/hr wage,
                  1 hr of rater setup). Surfaced, never hidden.
* **MODEL**     — derived from a model with explicit assumptions (e.g. an
                  estimated disagreement/adjudication rate).

We report cost in *labour-hours* and *$*
separately, and expose the nominal ratio only with an explicit caveat.

Usage
-----
    # Print a table to stdout using committed cost/timing files:
    python3 scripts/compute_cost.py

    # Write JSON (used by `make cost-report`):
    python3 scripts/compute_cost.py --output results/cost_breakdown.json

    # Override anything:
    python3 scripts/compute_cost.py \
        --n-responses 210 \
        --auto-sec 4.504 \
        --human-sec 12.2 \
        --disagreement 0.10 \
        --adj-sec 300 \
        --include-setup
"""

import argparse
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Defaults (kept importable so tests can call compute_breakdown)
# ──────────────────────────────────────────────────────────────
DEFAULT_N_RESPONSES = 210  # 105 prompts × 2 models
DEFAULT_AUTO_SEC_PER_PROMPT = 4.504  # MEASURED (results/cost_tracker.json avg/prompt)
DEFAULT_HUMAN_SEC_PER_LABEL = 12.2  # MEASURED (results/human_timing_measurement.json median)
DEFAULT_HUMAN_HR_RATE = 20.0  # ASSUMED working wage
DEFAULT_SETUP_HOURS = 1.0  # ASSUMED rater onboarding/training
DEFAULT_DISAGREEMENT_RATE = None  # not measured -> adjudication omitted, labelled
DEFAULT_ADJ_SEC_PER_CASE = 300.0  # ASSUMED adjudicator time per disagreement


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _measured_auto_sec() -> float:
    """Best-effort MEASURED auto seconds/prompt from cost_tracker.json."""
    p = Path("results/cost_tracker.json")
    if p.exists():
        try:
            return float(_load_json(p).get("avg_seconds_per_prompt", DEFAULT_AUTO_SEC_PER_PROMPT))
        except (OSError, ValueError):
            pass
    return DEFAULT_AUTO_SEC_PER_PROMPT


def _measured_human_sec() -> float:
    """Best-effort MEASURED human seconds/label from the timing study."""
    p = Path("results/human_timing_measurement.json")
    if p.exists():
        try:
            data = _load_json(p)
            val = data.get("median_seconds_per_label")
            if val:
                return float(val)
        except (OSError, ValueError):
            pass
    return DEFAULT_HUMAN_SEC_PER_LABEL


def compute_breakdown(
    n_responses: int = DEFAULT_N_RESPONSES,
    auto_seconds_per_prompt: float = DEFAULT_AUTO_SEC_PER_PROMPT,
    human_seconds_per_label: float = DEFAULT_HUMAN_SEC_PER_LABEL,
    human_hr_rate: float = DEFAULT_HUMAN_HR_RATE,
    disagreement_rate: float | None = None,
    adjudication_seconds_per_case: float = DEFAULT_ADJ_SEC_PER_CASE,
    include_setup: bool = True,
) -> dict:
    """Compute a full, explicitly-labelled cost breakdown.

    Separates MEASURED values from ASSUMED/MODEL values and computes auto, human
    and hybrid (50% audit) costs. The nominal auto-vs-human ratio is reported
    **only with an apples-to-oranges caveat** (labour vs ~free local compute).

    Args:
        n_responses: Total responses to evaluate.
        auto_seconds_per_prompt: MEASURED classifier seconds per prompt.
        human_seconds_per_label: MEASURED human seconds per label.
        human_hr_rate: ASSUMED hourly labour cost.
        disagreement_rate: Optional MODEL/disagreement proportion needing
            adjudication. ``None`` -> adjudication omitted & labelled as such.
        adjudication_seconds_per_case: ASSUMED seconds per adjudicated case.
        include_setup: Whether to include ASSUMED rater setup/training hours.

    Returns:
        dict with ``parameters``, ``MEASURED``, ``ASSUMED``/``MODEL``, per-scenario
        costs, and the caveated nominal ratio.
    """
    if n_responses <= 0:
        raise ValueError("n_responses must be positive")

    # ── AUTO ─────────────────────────────────────────────────
    auto_seconds = n_responses * auto_seconds_per_prompt
    auto_hours = auto_seconds / 3600
    # Local inference: ~free compute. Explicitly stated, NOT silently zero.
    auto_compute_rate_per_hr = 0.0  # local Ollama; no cloud cost (ASSUMED)
    auto_compute_cost = auto_hours * auto_compute_rate_per_hr

    # ── HUMAN ────────────────────────────────────────────────
    human_seconds = n_responses * human_seconds_per_label
    human_hours = human_seconds / 3600
    human_labor_cost = human_hours * human_hr_rate

    # Setup / training overhead (often significant, usually ignored). ASSUMED.
    setup_hours = DEFAULT_SETUP_HOURS if include_setup else 0.0
    setup_cost = setup_hours * human_hr_rate

    # Adjudication overhead. Only counted when a disagreement rate is supplied;
    # otherwise labelled as omitted (MODEL estimate absent → not silently 0).
    adjudication_cost = 0.0
    adjudication_label = "OMITTED (no measured disagreement rate)"
    if disagreement_rate is not None:
        n_disagreements = n_responses * disagreement_rate
        adjudication_hours = n_disagreements * adjudication_seconds_per_case / 3600
        adjudication_cost = adjudication_hours * human_hr_rate
        adjudication_label = (
            f"MODEL ({disagreement_rate:.0%} disagreement × "
            f"{adjudication_seconds_per_case:.0f}s/case, ASSUMED)"
        )

    total_human = human_labor_cost + setup_cost + adjudication_cost

    # ── HYBRID (50% auto + 50% human audit) ─────────────────
    audit_fraction = 0.5
    audit_responses = int(n_responses * audit_fraction)
    hybrid_auto_hours = (n_responses * auto_seconds_per_prompt) / 3600
    hybrid_human_hours = (audit_responses * human_seconds_per_label) / 3600
    hybrid_cost = hybrid_auto_hours * auto_compute_rate_per_hr + hybrid_human_hours * human_hr_rate

    # ── Nominal ratio, with an explicit apples-to-oranges caveat ─
    # Labour ÷ (≈free local compute) is not a universally-valid "× cheaper".
    ratio = total_human / max(auto_compute_cost, 1e-9) if auto_compute_cost > 0 else None

    return {
        "parameters": {
            "n_responses": n_responses,
            "auto_seconds_per_prompt": auto_seconds_per_prompt,
            "human_seconds_per_label": human_seconds_per_label,
            "disagreement_rate": disagreement_rate,
        },
        "MEASURED": {
            "auto_seconds_per_prompt": auto_seconds_per_prompt,
            "human_seconds_per_label": human_seconds_per_label,
        },
        "ASSUMED": {
            "human_hr_rate": human_hr_rate,
            "setup_hours": setup_hours,
            "setup_note": "Assumed rater onboarding/training; per-label times exclude it.",
            "auto_compute_rate_per_hr": auto_compute_rate_per_hr,
            "adjudication_seconds_per_case": (
                adjudication_seconds_per_case if disagreement_rate is not None else None
            ),
        },
        "MODEL": {
            "adjudication": adjudication_label,
        },
        "auto": {
            "hours": round(auto_hours, 3),
            "compute_cost_usd": round(auto_compute_cost, 3),
            "labour_hours": 0.0,
            "note": "Local inference ≈ $0 compute; 0 labour hours.",
        },
        "human": {
            "hours": round(human_hours, 3),
            "labour_cost_usd": round(human_labor_cost, 2),
            "setup_hours": setup_hours,
            "setup_cost_usd": round(setup_cost, 2),
            "adjudication_cost_usd": round(adjudication_cost, 2),
            "adjudication_label": adjudication_label,
            "total_cost_usd": round(total_human, 2),
        },
        "hybrid_50pct_audit": {
            "audit_responses": audit_responses,
            "auto_responses": n_responses,
            "cost_usd": round(hybrid_cost, 2),
            "note": "Run auto on all, human on 50% for validation.",
        },
        "comparison": {
            "human_labour_hours": round(human_hours, 3),
            "auto_labour_hours": 0.0,
            "labour_savings_hours": round(human_hours, 3),
            "nominal_ratio_x": round(ratio, 1) if ratio is not None else None,
            "caveat": (
                "Apples-to-oranges: the nominal ratio divides paid labour by "
                "≈free local compute. It is NOT a single universally-valid "
                "'× cheaper' claim — auto is preferable only where its κ is "
                "adequate for the dimension."
            ),
        },
    }


def _print_table(breakdown: dict) -> None:
    auto = breakdown["auto"]
    human = breakdown["human"]
    hybrid = breakdown["hybrid_50pct_audit"]
    comp = breakdown["comparison"]
    print(f"\nCost breakdown ({breakdown['parameters']['n_responses']} responses)\n")
    print(f"  Fully automatic : {auto['hours']:.2f} h compute, ~${auto['compute_cost_usd']:.2f}, 0 labour-h")
    print(
        f"  Fully human     : {human['hours']:.2f} h labour (~${human['labour_cost_usd']:.2f})"
        f" + setup ${human['setup_cost_usd']:.2f}"
        f" + adjudication ${human['adjudication_cost_usd']:.2f}"
        f" = ${human['total_cost_usd']:.2f}"
    )
    print(f"  Hybrid (50% aud.): ${hybrid['cost_usd']:.2f}")
    print(f"\n  Labour saved by auto: {comp['labour_savings_hours']:.2f} h")
    if comp["nominal_ratio_x"] is not None:
        print(f"  Nominal ratio (labour/compute): ~{comp['nominal_ratio_x']}×")
    print(f"  Caveat: {comp['caveat']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-responses", type=int, default=DEFAULT_N_RESPONSES)
    parser.add_argument("--auto-sec", type=float, default=None, help="Override auto s/prompt.")
    parser.add_argument("--human-sec", type=float, default=None, help="Override human s/label.")
    parser.add_argument("--rate", type=float, default=DEFAULT_HUMAN_HR_RATE, help="$/hr human wage.")
    parser.add_argument(
        "--disagreement", type=float, default=None, help="MODEL disagreement rate (0..1)."
    )
    parser.add_argument(
        "--adj-sec", type=float, default=DEFAULT_ADJ_SEC_PER_CASE, help="Assumed adjudicator s/case."
    )
    parser.add_argument("--no-setup", action="store_true", help="Exclude ASSUMED setup hours.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    auto_sec = args.auto_sec if args.auto_sec is not None else _measured_auto_sec()
    human_sec = args.human_sec if args.human_sec is not None else _measured_human_sec()

    breakdown = compute_breakdown(
        n_responses=args.n_responses,
        auto_seconds_per_prompt=auto_sec,
        human_seconds_per_label=human_sec,
        human_hr_rate=args.rate,
        disagreement_rate=args.disagreement,
        adjudication_seconds_per_case=args.adj_sec,
        include_setup=not args.no_setup,
    )

    _print_table(breakdown)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(breakdown, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
