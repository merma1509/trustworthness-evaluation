"""Reusable renderer for the blinded multi-rater re-annotation results.

This module does NOT own a top-level Streamlit tab. It exposes
``render_blinded_block(st)`` which renders the *content* of the blinded
re-validation — inter-annotator agreement, adjudication, and gold-vs-auto κ —
so that it can be embedded inside the "Human Annotation" tab.

It reads ``results/audit/inter_annotator_report.json`` (the output of
``scripts/run_blinded_annotation.py report``). When that file does not exist
(no ≥2 human annotators have run yet), it degrades gracefully to a clear
"pending re-annotation" notice rather than silently showing nothing.
"""
from pathlib import Path

import pandas as pd

from src.annotator import VALID_LABELS

REPORT_PATH = Path("results/audit/inter_annotator_report.json")


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


def render_blinded_block(st) -> None:
    """Render the blinded multi-rater re-validation content into a Streamlit app.

    Args:
        st: The Streamlit module (injected for testability).
    """
    st.markdown(
        "### Blinded Multi-Rater Re-Validation (recommended)"
    )
    st.markdown(
        "*≥2 independent **blinded** annotators; inter-rater κ is measured "
        "**before** any comparison to the auto-scorer.*"
    )

    if not REPORT_PATH.exists():
        st.warning(
            "**No multi-rater re-annotation yet.**\n\n"
            "Requires ≥2 independent human annotators to label the blinded calibration + "
            "held-out sets. This section will populate once the run has happened."
        )
        st.code(
            "make blinded-prepare            # emit per-annotator blinded templates\n"
            "# ... annotators fill human_label / confidence / notes ...\n"
            "make blinded-report DIMENSION=safety ANNOTATIONS=\"ann1.jsonl ann2.jsonl\"",
            language="bash",
        )
        st.info(
            "Working setup: the blinded templates carry **no** `auto_label` / similarity "
            "fields, so each annotator is independent of the auto-scorer."
        )
        return

    with REPORT_PATH.open() as f:
        report = __import__("json").load(f)

    # The report may be the new aggregated format (``by_dimension`` + ``overall``)
    # produced by the updated script, or the legacy single-dimension format.
    # Detect which one we have and render accordingly.
    by_dimension = report.get("by_dimension")
    is_aggregated = isinstance(by_dimension, dict) and len(by_dimension) > 0

    def _render_ia_block(ia: dict, title: str) -> None:
        st.markdown(f"#### {title}")
        if not ia:
            st.info("No inter-annotator data present.")
            return
        n_ann = ia.get("n_annotators", 0)
        mean_k = ia.get("mean_kappa", 0.0)
        mean_ag = ia.get("mean_agreement_rate", 0.0)
        annotator_names = ia.get("annotators", [])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Annotators", n_ann)
        with c2:
            st.metric("Mean pairwise κ", f"{mean_k:.3f}", help=_kappa_badge(mean_k))
        with c3:
            st.metric("Mean agreement", f"{mean_ag:.0%}")
        pairwise = ia.get("pairwise", {})
        if pairwise:
            rows = []
            for pair, stats in pairwise.items():
                rows.append({
                    "Annotator pair": pair.replace("__vs__", " ↔ "),
                    "n": stats.get("n", 0),
                    "Cohens κ": stats.get("cohens_kappa", 0.0),
                    "Agreement": stats.get("agreement_rate", 0.0),
                    "κ CI": (
                        f"[{stats.get('kappa_ci', {}).get('ci_lower', 0):.2f}, "
                        f"{stats.get('kappa_ci', {}).get('ci_upper', 0):.2f}]"
                        if stats.get("kappa_ci") else "—"
                    ),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if annotator_names:
            st.caption("Annotators: " + ", ".join(annotator_names))
        st.caption(
            "Quality gate: only trust the auto comparison below once inter-annotator "
            "κ is acceptable (e.g. ≥ 0.6 'Substantial')."
        )

    def _render_auto_block(auto_cmp: dict, title: str) -> None:
        st.markdown(f"#### {title}")
        if auto_cmp and auto_cmp.get("n_valid_pairs", 0):
            k = auto_cmp.get("cohens_kappa", 0.0)
            ag = auto_cmp.get("agreement_rate", 0.0)
            w = auto_cmp.get("weighted_kappa", 0.0)
            n = auto_cmp.get("n_valid_pairs", 0)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("n", n)
            with c2:
                st.metric("Cohen's κ", f"{k:.3f}", help=_kappa_badge(k))
            with c3:
                st.metric("Agreement", f"{ag:.0%}")
            with c4:
                st.metric("Weighted κ", f"{w:.3f}")
            ci = auto_cmp.get("kappa_ci")
            if ci:
                st.caption(
                    f"95% CI on κ: [{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}] "
                    f"(n_bootstrap={ci.get('n_bootstrap', '—')})"
                )
        else:
            st.info("No auto comparison available yet — run the report after annotators finish.")

    if is_aggregated:
        # ---- NEW aggregated format: show per-dimension summary first ----
        st.markdown("#### Inter-Annotator Agreement by Dimension (the gate)")
        per_dim_rows = []
        for dim, sub in sorted(by_dimension.items()):
            ia = sub.get("inter_annotator", {})
            ac = sub.get("auto_comparison", {})
            per_dim_rows.append({
                "Dimension": dim,
                "Annotators n": ia.get("n_common_annotations_total", 0),
                "Inter κ": ia.get("mean_kappa", 0.0),
                "Inter agreement": ia.get("mean_agreement_rate", 0.0),
                "Gold n": ac.get("n_valid_pairs", 0),
                "Gold vs auto κ": ac.get("cohens_kappa", 0.0),
                "Gold vs auto agreement": ac.get("agreement_rate", 0.0),
            })
        st.dataframe(pd.DataFrame(per_dim_rows), width="stretch", hide_index=True)

        # Overall pooled block
        overall = report.get("overall", {})
        st.markdown("### Overall (pooled across dimensions)")
        _render_ia_block(overall.get("inter_annotator", {}), "Inter-Annotator Agreement")
        adj = overall.get("adjudicated", {})
        recs = adj.get("records", [])
        n_total = adj.get("n_total", len(recs))
        n_adjudicate = adj.get("n_needs_adjudication", 0)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Gold records", n_total)
        with c2:
            st.metric("Needs human adjudication", n_adjudicate)
        if n_adjudicate:
            st.warning(
                f"{n_adjudicate} record(s) had a tie with no majority — these need "
                "human adjudication before the auto comparison is final."
            )
        _render_auto_block(overall.get("auto_comparison", {}),
                           "Adjudicated Gold vs Auto-Scorer (headline κ)")

        # Per-dimension auto comparison details
        st.markdown("#### Per-Dimension Gold vs Auto Scorer")
        for dim, sub in sorted(by_dimension.items()):
            _render_auto_block(
                sub.get("auto_comparison", {}),
                f"{dim.capitalize()} — Gold vs Auto",
            )
    else:
        # ---- LEGACY single-dimension format (backwards compatibility) ----
        ia = report.get("inter_annotator", {})
        _render_ia_block(ia, "Inter-Annotator Agreement (the gate)")
        adjudicated = report.get("adjudicated", {})
        st.markdown("#### Adjudication")
        if adjudicated:
            recs = adjudicated.get("records", [])
            n_total = adjudicated.get("n_total", len(recs))
            n_adjudicate = adjudicated.get("n_needs_adjudication", 0)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Gold records", n_total)
            with c2:
                st.metric("Needs human adjudication", n_adjudicate)
            if n_adjudicate:
                st.warning(
                    f"{n_adjudicate} record(s) had a tie with no majority — these need "
                    "human adjudication before the auto comparison is final."
                )
        else:
            st.info("No adjudication data present in report.")
        _render_auto_block(report.get("auto_comparison", {}),
                           "Adjudicated Gold vs Auto-Scorer (headline κ)")

    # ── Rubric / labels reminder ──────────────────────────
    st.markdown("#### Expected labels (per dimension)")
    label_df = pd.DataFrame([
        {"Dimension": dim, "Allowed labels": " / ".join(v)}
        for dim, v in sorted(VALID_LABELS.items())
    ])
    st.dataframe(label_df, width="stretch", hide_index=True)
