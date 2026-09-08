"""Reusable renderer for the sealed multi-rater re-annotation results.

This module does NOT own a top-level Streamlit tab. It exposes
``render_blinded_block(st)`` which renders the *content* of the sealed
re-validation — inter-rater agreement (RATER_A vs RATER_B) and the gold-vs-auto
κ — so that it can be embedded inside the "Human Annotation" tab.

It reads ``experiment/reports/part1_agreement_report.json`` (the output of
``scripts/report_part1_agreement.py`` on the sealed CLEAN-REDO experiment flow).
When that file does not exist (no sealed annotations have been produced yet), it
degrades gracefully to a clear "pending re-annotation" notice rather than
silently showing nothing.
"""

import json
from pathlib import Path

import pandas as pd

from src.labels import VALID_LABELS

# Sealed CLEAN-REDO experiment report (the single agreement report consumed by
# the dashboard). Produced by scripts/report_part1_agreement.py.
SEALED_REPORT_PATH = Path("experiment/reports/part1_agreement_report.json")

# Canonical dimension order used when pooling (matches src/labels.DIMENSIONS).
DIMENSIONS = ("safety", "truthfulness", "consistency")


def _pick_report_path() -> Path | None:
    """Return the sealed report path if it exists, else None."""
    return SEALED_REPORT_PATH if SEALED_REPORT_PATH.exists() else None


def _kappa_badge(k: float) -> str:
    """Qualitative interpretation of Cohen's κ, matching the rubric."""
    if k >= 0.81:
        return "Almost perfect"
    if k >= 0.61:
        return "Substantial"
    if k >= 0.41:
        return "Moderate"
    if k >= 0.21:
        return "Fair"
    if k >= 0.0:
        return "Slight"
    return "Poor (worse than random)"


def _render_rater_block(st, by_dim: dict, title: str) -> None:
    """Render the inter-rater (RATER_A vs RATER_B) agreement per dimension.

    Args:
        st: The Streamlit module (injected for testability).
        by_dim: The ``inter_rater_A_vs_B`` dict from the sealed report.
        title: Section heading.
    """
    st.markdown(f"#### {title}")
    if not by_dim:
        st.info("No inter-rater data present.")
        return

    # Pool the available dimensions so the headline gate is shown up-front.
    rows = []
    for dim in DIMENSIONS:
        s = by_dim.get(dim) or {}
        if not s:
            continue
        rows.append(
            {
                "Dimension": dim,
                "n": s.get("n", 0),
                "Cohens κ": s.get("cohens_kappa", 0.0),
                "Agreement": s.get("agreement_rate", 0.0),
            }
        )

    if rows:
        c1, c2 = st.columns(2)
        mean_k = sum(r["Cohens κ"] for r in rows) / len(rows)
        best_k = max(r["Cohens κ"] for r in rows)
        with c1:
            st.metric("Mean dimension κ", f"{mean_k:.3f}", help=_kappa_badge(mean_k))
        with c2:
            st.metric("Best dimension κ", f"{best_k:.3f}", help=_kappa_badge(best_k))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "RATER_A vs RATER_B inter-rater agreement — the independent-rater gate "
        "measured *before* any comparison to the auto-scorer."
    )


def _render_gold_vs_auto_block(st, gva: dict, title: str) -> None:
    """Render the gold (adjudicated human) vs auto-scorer agreement.

    Args:
        st: The Streamlit module (injected for testability).
        gva: The ``gold_vs_auto`` dict from the sealed report.
        title: Section heading.
    """
    st.markdown(f"#### {title}")

    overall = (gva or {}).get("overall") or {}
    per_dim = (gva or {}).get("per_dimension") or {}

    if overall:
        n = overall.get("n", 0) or overall.get("n_valid_pairs", 0)
        k = overall.get("cohens_kappa", 0.0)
        ag = overall.get("agreement_rate", 0.0)
        w = overall.get("weighted_kappa", 0.0)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("n", n)
        with c2:
            st.metric("Cohen's κ", f"{k:.3f}", help=_kappa_badge(k))
        with c3:
            st.metric("Agreement", f"{ag:.0%}")
        with c4:
            st.metric("Weighted κ", f"{w:.3f}")
        ci = overall.get("kappa_ci")
        if ci:
            st.caption(
                f"95% CI on κ: [{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}] "
                f"(n_bootstrap={ci.get('n_bootstrap', '—')})"
            )
    else:
        st.info("No overall gold-vs-auto data present.")

    # Per-dimension table (gold vs auto) — the headline κ per dimension.
    if per_dim:
        st.markdown("#### Per-Dimension Gold vs Auto Scorer")
        rows = []
        for dim in DIMENSIONS:
            s = per_dim.get(dim) or {}
            if not s:
                continue
            ci = s.get("kappa_ci") or {}
            rows.append(
                {
                    "Dimension": dim,
                    "n": s.get("n", 0) or s.get("n_valid_pairs", 0),
                    "Cohen's κ": s.get("cohens_kappa", 0.0),
                    "Agreement": s.get("agreement_rate", 0.0),
                    "κ CI": (
                        f"[{ci.get('ci_lower', 0):.2f}, {ci.get('ci_upper', 0):.2f}]"
                        if ci
                        else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_blinded_block(st) -> None:
    """Render the sealed multi-rater re-validation content into a Streamlit app.

    Args:
        st: The Streamlit module (injected for testability).
    """
    st.markdown("### Sealed Multi-Rater Re-Validation (CLEAN-REDO)")
    st.markdown(
        "*Independent raters annotate sealed templates with a constrained label "
        "vocabulary; inter-rater κ (RATER_A vs RATER_B) is measured **before** any "
        "comparison to the sealed auto-scorer.*"
    )

    if not SEALED_REPORT_PATH.exists():
        st.warning(
            "**No sealed multi-rater re-annotation yet.**\n\n"
            "Requires independent raters to fill the sealed templates and the "
            "agreement report to be produced. This section will populate once the "
            "run has happened."
        )
        st.code(
            'make experiment-seal SEED=... EXPERIMENT_ID=...  # seal a NEW experiment\n'
            'make experiment-annotate RATERS="raterA raterB"  # emit blank rater templates\n'
            "# ... raters fill experiment/annotations/*.jsonl (label-constrained) ...\n"
            'make experiment-gold SEAL_PASSPHRASE=...         # build gold labels\n'
            "make experiment-agreement                         # held-out agreement κ\n"
            "make experiment-reproduce                         # full reproduction (honesty gate)",
            language="bash",
        )
        st.info(
            "Sealed templates carry only the prompt + dimension (no auto label / "
            "model identity), so each rater is independent of the auto-scorer."
        )
        return

    report_path = _pick_report_path()
    with report_path.open() as f:
        report = json.load(f)
    st.caption(
        "Source: sealed CLEAN-REDO experiment "
        "(`experiment/reports/part1_agreement_report.json`)."
    )

    gva = report.get("gold_vs_auto") or {}
    inter_rater = report.get("inter_rater_A_vs_B") or {}

    # Inter-rater gate first — independent agreement before gold-vs-auto.
    _render_rater_block(st, inter_rater, "Inter-Rater Agreement (RATER_A vs RATER_B)")

    st.markdown("---")
    _render_gold_vs_auto_block(st, gva, "Adjudicated Gold vs Auto-Scorer (headline κ)")

    # ── Rubric / labels reminder ──────────────────────────
    st.markdown("#### Expected labels (per dimension)")
    label_df = pd.DataFrame(
        [
            {"Dimension": dim, "Allowed labels": " / ".join(sorted(v))}
            for dim, v in sorted(VALID_LABELS.items())
        ]
    )
    st.dataframe(label_df, width="stretch", hide_index=True)
