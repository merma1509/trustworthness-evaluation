#!/usr/bin/env python3
"""Instrument pipeline-loop diagram.

A purely structural diagram of the *measurement-validation* loop. It carries
NO numeric values on purpose: every `make run` produces fresh data, so the
figure must never hardcode agreement scores, costs, or sample sizes. The boxes
/ arrows stay constant while the numbers behind them (shown in the dashboard
and reports) change per run.

Flow (left-to-right with a feedback loop):

    Auto-Scorer ─► Calibration split ─► Reliability estimate ─► κ-gate? ─► (score)
         ▲                                                         │
         └──────────── Human-in-the-loop re-annotation ◄───────────┘

Branches at the κ gate (values 0.7/0.4 are
deliberately omitted here, see `make experiment-budget` / dashboard for bands):
  - high agreement  -> direct use for rankings & score card
  - low agreement   -> flag Unverified, route to human annotation (feedback loop)

The single hardcoded element is the *gate labels* ("Trust" / "Caveated" /
"Don't trust"), which are band names, not numbers.

Output: results/pipeline_loop.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RESULTS_DIR


def draw_pipeline_loop(out_path: Path) -> None:
    """Render the pipeline-loop diagram with matplotlib patches."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print(" matplotlib not installed. Skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # ── helpers ────────────────────────────────────────────
    def hbox(x, y, w, h, text, fc="#eaf3fb", ec="#1f6fb2", fs=11):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fc,
            ec=ec,
            lw=1.8,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color="#123456",
        )
        return (x + w, y + h)

    def arrow(x1, y1, x2, y2, color="#444", lw=2.2, style="-|>"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle="arc3,rad=0"),  # noqa: RUF100
        )

    title = "Measurement-Validation Loop\n"
    title += "(profit/κ/cost values shown in reports & dashboard — refreshed each run)"
    ax.text(
        6, 5.85, title, ha="center", va="center", fontsize=14, fontweight="bold", color="#123456"
    )

    # ── main flow (top row, left -> right) ─────────────────
    bx1, by1, bw, bh = 0.4, 3.2, 2.2, 1.1
    hbox(bx1, by1, bw, bh, "Auto-Scorer\n(local, deterministic)")
    x2, _ = hbox(3.1, 3.2, 2.2, 1.1, "Calibration /\nHeld-out split")
    arrow(bx1 + bw, 3.75, 3.1, 3.75)
    arrow(x2, 3.75, 5.6, 3.75)
    hbox(5.6, 3.2, 2.4, 1.1, "Reliability\nestimate (per-dimension κ)")
    arrow(8.0, 3.75, 8.6, 3.75)
    gate_x, gate_y, gate_w, gate_h = 8.6, 3.2, 2.1, 1.1
    hbox(gate_x, gate_y, gate_w, gate_h, "κ-gate\n(decision)", fc="#fef3d9", ec="#b8860b")

    # ── score output (top right) ──────────────────────────
    hbox(11.0, 3.2, 0.0, 0.0, "")  # no-op, reserved
    arrow(10.7, 3.75, 11.2, 3.75)
    hbox(11.2, 3.2, 0.6, 1.1, "Score\ncard", fc="#eafbea", ec="#2e7d32")

    # ── gate branches (right -> bottom) ────────────────────
    # Trust (high κ) -> direct to score card
    arrow(gate_x + gate_w, 4.5, 11.2, 4.5, color="#2e7d32")
    ax.text(
        10.0, 4.75, "Trust  (κ high)", ha="center", fontsize=9, color="#2e7d32", fontstyle="italic"
    )
    # Caveated / Don't-trust (low κ) -> human re-annotation loop
    arrow(gate_x + 1.05, gate_y, gate_x + 1.05, 1.4, color="#c0392b")
    ax.text(
        gate_x + 1.15,
        2.35,
        "Caveated / Don't trust\n(κ low -> route to humans)",
        ha="left",
        fontsize=9,
        color="#c0392b",
    )
    hbox(0.4, 1.4, 2.4, 1.1, "Human-in-the-loop\nre-annotation", fc="#fdecea", ec="#c0392b")
    arrow(2.8, 1.95, 4.2, 1.95)
    hbox(4.2, 1.4, 2.3, 1.1, "Blinded\nre-label")
    # feedback loop: human labels -> re-estimate reliability
    arrow(6.5, 1.95, 6.5, 3.2, color="#1f6fb2", style="-|>")
    arrow(4.2, 1.95, 3.0, 1.95)
    ax.text(
        3.1,
        1.15,
        "feedback: measures κ, feeds budget policy",
        ha="left",
        fontsize=9,
        color="#1f6fb2",
        fontstyle="italic",
    )

    # ── legend box (bands, not numbers) ───────────────────
    legend_y = 5.55
    ax.text(1.2, legend_y, "Trust region bands", fontsize=10, fontweight="bold", color="#123456")
    bands = [("Trust", "#2e7d32"), ("Caveated", "#b8860b"), ("Don't trust", "#c0392b")]
    for i, (name, col) in enumerate(bands):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (1.2 + i * 1.3, legend_y - 0.35),
                1.0,
                0.28,
                boxstyle="round,pad=0.01",
                fc=col,
                ec="none",
                alpha=0.85,
            )
        )
        ax.text(
            1.7 + i * 1.3,
            legend_y - 0.21,
            f"κ ≥? {name}",
            fontsize=8,
            color="white",
            ha="center",
            va="center",
        )

    # Footer: where the band threshold values live (not hardcoded here)
    ax.text(
        6,
        0.5,
        "Band thresholds (0.7 / 0.4 etc.) are runtime params — "
        "see dashboard 'Budget Optimization' & validation report.",
        ha="center",
        fontsize=9,
        color="#666",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RESULTS_DIR / "pipeline_loop.png"))
    args = parser.parse_args()
    draw_pipeline_loop(Path(args.output))


if __name__ == "__main__":
    main()

