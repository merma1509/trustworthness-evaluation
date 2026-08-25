#!/usr/bin/env python3
"""Error Heatmap.

Visualises the auto-vs-human agreement per dimension and the pooled
Auto × Human confusion matrix, making the systematic error cells explicit
(e.g. hallucination flagged ``correct``, over-refusal on Safety).

Two panels:

1. **Per-dimension agreement matrix** — rows are dimensions, columns are the
   agreement metrics (sample size, agreement %, Cohen's κ, per-label precision).
   This surfaces *where* the scorer is reliable (green) vs *where* it collapses
   (red, e.g. Truthfulness at chance level).

2. **Auto × Human confusion heatmap** — the pooled confusion matrix (rows =
   auto label, cols = human label) colour-scaled by count, with the diagonal
   (agreements) separated from the off-diagonal (systematic disagreements).

Sources (any existing):
  * ``results/audit/agreement_report.json`` — overall/by_dimension agreement + CM
  * ``results/validation_report.json`` — RQ1 by_dimension κ

Output: results/error_heatmap.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RESULTS_DIR

DIMENSIONS = ["safety", "truthfulness", "consistency"]
DIMENSION_LABELS = {
    "safety": "Safety",
    "truthfulness": "Truthfulness",
    "consistency": "Consistency",
}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _build_dimension_matrix(agreement_report: dict, validation_report: dict) -> list:
    """Return per-dimension rows of {'dimension', 'n', 'agreement', 'kappa', 'precision'}."""
    bd_a = (agreement_report or {}).get("by_dimension") or {}
    bd_v = ((validation_report or {}).get("rq1_agreement") or {}).get("by_dimension") or {}

    rows = []
    for dim in DIMENSIONS:
        a = bd_a.get(dim) or {}
        v = bd_v.get(dim) or {}
        n = a.get("n", a.get("n_valid_pairs", 0))
        agreement = a.get("agreement_rate", 0)
        kappa = v.get("cohens_kappa", a.get("cohens_kappa", 0))
        # Per-label precision of 'correct' (or 'consistent') when available.
        pl = (a.get("per_label_agreement") or {}).get("correct")
        if pl is None:
            pl = (a.get("per_label_agreement") or {}).get("consistent")
        precision = pl.get("precision", None) if pl else None
        rows.append(
            {
                "dimension": DIMENSION_LABELS.get(dim, dim),
                "n": n,
                "agreement": agreement,
                "kappa": kappa,
                "precision": precision,
            }
        )
    return rows


def _build_confusion_matrix(agreement_report: dict) -> tuple:
    """Return (labels, matrix) for the pooled Auto × Human CM, or (None, None)."""
    cm = (agreement_report or {}).get("overall", {}).get("confusion_matrix")
    if not cm:
        return None, None
    # Union of labels that appear in either rows or columns.
    labels = []
    for row_label, col_map in cm.items():
        if row_label not in labels:
            labels.append(row_label)
        for col_label in col_map or {}:
            if col_label not in labels:
                labels.append(col_label)
    labels.sort()
    matrix = [[cm.get(r, {}).get(c, 0) for c in labels] for r in labels]
    return labels, matrix


def plot_error_heatmap(agreement_path: Path, validation_path: Path, out_path: Path) -> None:
    """Render the two-panel error heatmap PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(" matplotlib not installed. Skipping figure.")
        return

    agreement_report = _load(agreement_path) if agreement_path.exists() else {}
    validation_report = _load(validation_path) if validation_path.exists() else {}
    matrix_rows = _build_dimension_matrix(agreement_report, validation_report)

    labels, cm = _build_confusion_matrix(agreement_report)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: per-dimension agreement matrix ─────────────
    ax = axes[0]
    if matrix_rows:
        metric_cols = ["N", "Agreement", "κ", "Precision"]
        data = []
        for r in matrix_rows:
            data.append(
                [
                    r["n"],
                    r["agreement"],
                    r["kappa"],
                    r["precision"] if r["precision"] is not None else 0.0,
                ]
            )
        import numpy as np

        arr = np.array(data, dtype=float)
        # Normalise each column independently for colour scale.
        norm = arr.copy()
        for col in range(arr.shape[1]):
            col_vals = arr[:, col]
            lo, hi = col_vals.min(), col_vals.max()
            norm[:, col] = (col_vals - lo) / (hi - lo) if hi > lo else 0.0
        im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_cols)
        ax.set_yticks(range(len(matrix_rows)))
        ax.set_yticklabels([r["dimension"] for r in matrix_rows])
        # Annotate actual values.
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                text = (
                    (f"{val:.0f}" if j == 0 else f"{val:.0%}" if j >= 1 else f"{val:.2f}")
                    if j != 2
                    else f"{val:.2f}"
                )
                ax.text(j, i, text, ha="center", va="center", fontsize=10, color="black")
        ax.set_title("Per-dimension agreement (normalised colour)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "No agreement report", ha="center", va="center")
        ax.axis("off")

    # ── Panel 2: pooled Auto × Human confusion ──────────────
    ax = axes[1]
    if cm is not None and labels:
        import numpy as np

        arr = np.array(cm, dtype=float)
        im = ax.imshow(arr, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Human label")
        ax.set_ylabel("Auto label")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{int(arr[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if arr[i, j] > 8e-5 else "black",
                )
        ax.set_title("Pooled Auto × Human confusion (counts)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
        ax.axis("off")

    fig.suptitle("Error Heatmap — where the auto-scorer and humans disagree", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", default=str(RESULTS_DIR / "audit/agreement_report.json"))
    parser.add_argument("--validation", default=str(RESULTS_DIR / "validation_report.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "error_heatmap.png"))
    args = parser.parse_args()

    plot_error_heatmap(Path(args.agreement), Path(args.validation), Path(args.output))


if __name__ == "__main__":
    main()
