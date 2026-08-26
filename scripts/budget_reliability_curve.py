#!/usr/bin/env python3
"""Budget vs Reliability curve (Part-3 Figure 3).

Illustrates the central idea of the paper's budget optimizer: human effort should
be spent *where* it buys the most reliability, and it does not buy reliability
uniformly across dimensions.

For each dimension we plot the *reliability gain* (measured as the improvement in
the auto-vs-gold Cohen's κ) as a function of the human labels spent. The shape of
each curve is dimension-specific:

* a **robust** dimension (auto already agrees strongly with humans) has *low
  marginal value* from extra human eyes  — the curve is flat / diminishing;
* a **brittle** dimension (auto near chance) has *high marginal value* — the curve
  rises steeply as human labels replace uncertain auto decisions.

We model reliability proxyatively from the calibration κ per dimension (from
``results/validation_report.json`` RQ1 ``by_dimension``, or an explicit
``--kappas`` argument). The N axis uses the per-dimension record counts; cost is
derived from the measured human label rate.

Output: results/budget_reliability_curve.png (and prints the table).

Usage
-----
    python3 scripts/budget_reliability_curve.py
    python3 scripts/budget_reliability_curve.py --report results/validation_report.json
    python3 scripts/budget_reliability_curve.py --kappas safety=0.62 truthfulness=0.0 consistency=0.62
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RESULTS_DIR

# Measured human-label economics (Part 1, rq4_cost).
HUMAN_SECONDS_PER_LABEL = 8.01
HUMAN_HOURLY_COST = 20.0

# Per-dimension total record counts on the full audit set.
DEFAULT_N = {
    "safety": 70,  # 35 prompts x 2 models
    "truthfulness": 76,  # 38 prompts x 2 models
    "consistency": 64,  # 32 prompts x 2 models (paired groups scaled)
}
DIMENSION_LABELS = {
    "safety": "Safety",
    "truthfulness": "Truthfulness",
    "consistency": "Consistency",
}
COLORS = {
    "safety": "#3498db",
    "truthfulness": "#e74c3c",
    "consistency": "#2ecc71",
}


def _load_kappas_from_report(report_path: Path) -> dict:
    """Read per-dimension κ from either report schema (validation or experiment)."""
    with report_path.open() as f:
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


def _parse_kappas(pairs_str) -> dict:
    out = {}
    for item in pairs_str:
        dim, _, val = item.partition("=")
        try:
            out[dim.strip()] = float(val)
        except ValueError:
            continue
    return out


def reliability_gain(kappa: float, labels_spent: int, total_n: int, k_sat: float = 0.25) -> float:
    """Proxy for reliability as a function of human labels spent.

    Models the auto-vs-gold agreement lifting from ``kappa`` toward a cap as
    humans label an increasing share of the dimension, with *diminishing
    returns*: each extra label buys less than the previous one.

        gain(share) = kappa + headroom * share / (share + k_sat)

    where ``share = labels_spent / total_n`` and ``headroom = 1 - kappa``.
    ``k_sat`` is the share at which half the headroom is captured (smaller =
    faster to approach the cap).

    * Robust dimensions (high κ) have a small headroom → their curve stays
      flat (little to gain).
    * Brittle dimensions (low κ, e.g. near-chance Truthfulness) have a large
      headroom → steep early gain as humans replace uncertain auto decisions.

    Args:
        kappa: Starting auto-vs-gold κ (0..1).
        labels_spent: Number of human labels allocated.
        total_n: Total records in the dimension.
        k_sat: Saturation rate constant (>0) controlling how quickly the curve
            approaches the cap. Smaller = faster saturation.

    Returns:
        Reliability measure in [kappa, 1.0].
    """
    share = labels_spent / max(total_n, 1)
    headroom = 1.0 - kappa
    return kappa + headroom * share / (share + k_sat)


def build_curves(kappas: dict, ns: dict = None) -> dict:
    """Compute reliability curves for all dimensions.

    Args:
        kappas: ``{dimension: kappa}``.
        ns: Optional ``{dimension: total_n}`` override.

    Returns:
        ``{dimension: {"labels": [...], "reliability": [...], "cost": [...]}}``.
    """
    ns = ns or DEFAULT_N
    curves = {}
    for dim, k in kappas.items():
        total_n = ns.get(dim, 50)
        labels = list(range(0, total_n + 1, max(1, total_n // 30)))
        rel = [reliability_gain(k, lab, total_n) for lab in labels]
        cost = [
            round(lab * HUMAN_SECONDS_PER_LABEL / 3600 * HUMAN_HOURLY_COST, 2) for lab in labels
        ]
        curves[dim] = {
            "labels": labels,
            "reliability": rel,
            "cost": cost,
            "kappa": k,
            "total_n": total_n,
        }
    return curves


def plot_curves(curves: dict, out_path: Path) -> None:
    """Render the reliability-vs-budget curves to a PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(" matplotlib not installed. Skipping figure; printing table instead.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for dim, c in curves.items():
        ax.plot(
            c["labels"],
            c["reliability"],
            color=COLORS.get(dim, "#888888"),
            linewidth=2.5,
            label=(
                f"{DIMENSION_LABELS.get(dim, dim)} (κ start = {c['kappa']:.2f}, n = {c['total_n']})"
            ),
        )

    ax.axhline(
        0.7, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Trust gate (κ=0.7)"
    )
    ax.set_xlabel("Human labels spent", fontsize=12)
    ax.set_ylabel("Auto-vs-gold reliability (proxy κ)", fontsize=12)
    ax.set_title("Budget vs Reliability: where does human effort buy the most?", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="", help="Optional report JSON to source per-dimension κ from."
    )
    parser.add_argument(
        "--kappas",
        nargs="*",
        default=[],
        help="Inline κ overrides, e.g. safety=0.62 truthfulness=0.0",
    )
    parser.add_argument("--output", default=str(RESULTS_DIR / "budget_reliability_curve.png"))
    args = parser.parse_args()

    # Resolve per-dimension κ: inline overrides win; fall back to report; else defaults.
    kappas = {}
    if args.report and Path(args.report).exists():
        kappas.update(_load_kappas_from_report(Path(args.report)))
    inline = _parse_kappas(args.kappas)
    kappas.update(inline)
    # Calibration defaults when nothing is provided.
    defaults = {"safety": 0.615, "truthfulness": 0.0, "consistency": 0.615}
    kappas = {
        d: kappas.get(d, defaults.get(d, 0.5)) for d in ("safety", "truthfulness", "consistency")
    }

    curves = build_curves(kappas)
    plot_curves(curves, Path(args.output))

    # Always print the table for text use.
    print("\nReliability lifted by human labels (proxy κ):")
    header = f"{'labels':>6}  " + "  ".join(f"{DIMENSION_LABELS[d]:>14}" for d in kappas)
    print(header)
    for idx in range(0, 21):
        lab = idx * 3
        row = f"{lab:>6}  "
        for d in kappas:
            c = curves[d]
            val = reliability_gain(c["kappa"], lab, c["total_n"])
            row += f"{val:>14.3f}"
        print(row)
    print(f"\nStarting κ: {', '.join(f'{d}={k:.3f}' for d, k in kappas.items())}")


if __name__ == "__main__":
    main()
