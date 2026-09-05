#!/usr/bin/env python3
"""Cost-vs-reliability frontier per dimension

The proper way to express "budget optimization" is a **cost vs reliability
frontier**, not a single claimed-optimal number. For each dimension we can choose
the fraction (0→1) of prompts annotated by humans. Reliability (agreement with
truth) improves as more human annotation is added, while cost rises linearly.

IMPORTANT — this is a **MODEL ESTIMATE**, not a measurement:

* Human accuracy is assumed (≈0.9 from prior studies) — explicitly labelled.
* The effective-agreement formula is a simple convex mix of auto κ and assumed
  human accuracy. It is a modelling choice, not an observed curve.
* The auto κ inputs ARE measured (from the validated/report), but their
  extrapolation to "reliability at fraction human" is a model.

The numbers here must never be presented as measured results; they are a
decision-support illustration only.

Usage
-----
    python3 scripts/cost_reliability_frontier.py
    python3 scripts/cost_reliability_frontier.py --report results/validation_report.json
    python3 scripts/cost_reliability_frontier.py --kappas safety=0.593 truthfulness=0.277 consistency=0.0
    python3 scripts/cost_reliability_frontier.py --output results/cost_reliability_frontier.json
"""

import argparse
import json
from pathlib import Path

# Per-dimension independent-unit counts (record counts on the full audit set).
# NOTE: these are the *annotatable* counts; safety/truthfulness are per-prompt,
# consistency is per-group. Used only to convert fraction→absolute labels for
# cost, so this is an ASSUMED/MODEL aid, not a headline.
DEFAULT_N = {
    "safety": 70,
    "truthfulness": 76,
    "consistency": 64,
}
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

HUMAN_SECONDS_PER_LABEL = 12.2  # MEASURED median from the timing study
HUMAN_HOURLY_COST = 20.0  # ASSUMED wage
HUMAN_ACCURACY_ASSUMED = 0.9  # ASSUMED human accuracy (explicit modelling choice)


def _load_report_kappas(path: Path) -> dict:
    """Read per-dimension auto-vs-gold κ from either report schema."""
    with path.open() as f:
        report = json.load(f)
    out: dict = {}
    bd = report.get("by_dimension")
    if isinstance(bd, dict):
        for dim, sub in bd.items():
            ac = (sub or {}).get("auto_comparison") or {}
            k = ac.get("cohens_kappa")
            if k is not None:
                out[dim] = k
        if out:
            return out
    rq1 = report.get("rq1_agreement") or {}
    bd2 = rq1.get("by_dimension")
    if isinstance(bd2, dict):
        for dim, sub in bd2.items():
            k = (sub or {}).get("cohens_kappa")
            if k is not None:
                out[dim] = k
    return out


def _parse_kappas(pairs) -> dict:
    out = {}
    for item in pairs:
        dim, _, val = item.partition("=")
        try:
            out[dim.strip()] = float(val)
        except ValueError:
            continue
    return out


def build_frontier(
    kappas: dict,
    ns: dict | None = None,
    human_accuracy: float = HUMAN_ACCURACY_ASSUMED,
    seconds_per_label: float = HUMAN_SECONDS_PER_LABEL,
    hourly_cost: float = HUMAN_HOURLY_COST,
) -> dict:
    """Build the cost-reliability curve for each dimension.

    Args:
        kappas: ``{dimension: auto_vs_gold_kappa}`` (MEASURED inputs).
        ns: Optional ``{dimension: n_responses}``.
        human_accuracy: ASSUMED human accuracy (MODEL assumption).
        seconds_per_label: MEASURED human seconds per label.
        hourly_cost: ASSUMED $/hr.

    Returns:
        dict with per-dimension frontier points labelled as MODEL ESTIMATE.
    """
    ns = ns or DEFAULT_N
    frontiers = {}
    for dim, kappa in kappas.items():
        n_prompts = ns.get(dim, 50)
        full_cost = n_prompts * seconds_per_label / 3600 * hourly_cost
        points = []
        for frac in FRACTIONS:
            cost = full_cost * frac
            # MODEL: effective agreement = convex mix of assumed human accuracy
            # and measured auto κ. Explicitly a modelling assumption.
            effective = frac * human_accuracy + (1 - frac) * max(kappa, 0.0)
            points.append(
                {
                    "frac_human": frac,
                    "human_labels": round(n_prompts * frac),
                    "cost_usd": round(cost, 2),
                    "effective_agreement_model": round(effective, 3),
                    "label": "MODEL ESTIMATE (not measured)",
                }
            )
        frontiers[dim] = {
            "kappa_auto_measured": round(kappa, 3),
            "n_responses": n_prompts,
            "human_accuracy_assumed": human_accuracy,
            "seconds_per_label_measured": seconds_per_label,
            "points": points,
        }
    return {
        "label": "MODEL ESTIMATE — cost-reliability frontier. Human accuracy "
        "assumed, not measured. Not a validated result.",
        "human_accuracy_assumed": human_accuracy,
        "frontiers": frontiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="", help="Optional report JSON to source measured κ from."
    )
    parser.add_argument(
        "--kappas",
        nargs="*",
        default=[],
        help="Inline κ, e.g. safety=0.593 truthfulness=0.277 consistency=0.0",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path for the frontier.")
    parser.add_argument(
        "--human-accuracy",
        type=float,
        default=HUMAN_ACCURACY_ASSUMED,
        help="ASSUMED human accuracy used in the model.",
    )
    args = parser.parse_args()

    kappas = {}
    if args.report and Path(args.report).exists():
        kappas.update(_load_report_kappas(Path(args.report)))
    kappas.update(_parse_kappas(args.kappas))
    defaults = {"safety": 0.593, "truthfulness": 0.277, "consistency": 0.0}
    kappas = {
        d: kappas.get(d, defaults.get(d, 0.5)) for d in ("safety", "truthfulness", "consistency")
    }

    frontier = build_frontier(kappas, human_accuracy=args.human_accuracy)

    print(f"\n{frontier['label']}\n")
    for dim, f in frontier["frontiers"].items():
        print(
            f"{dim.capitalize():<13} (auto κ={f['kappa_auto_measured']:.3f}, n={f['n_responses']}):"
        )
        for p in f["points"]:
            print(
                f"    {p['frac_human'] * 100:>4.0f}% human  labels={p['human_labels']:>4}  "
                f"cost=${p['cost_usd']:>6.2f}  eff.agreement={p['effective_agreement_model']:.3f}"
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fobj:
            json.dump(frontier, fobj, indent=2, ensure_ascii=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
