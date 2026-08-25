"""Tab: Human Annotation — validation of auto-labels against human judgment.

Merged view (:
    1. **Blinded multi-rater re-validation** (recommended, shown first) — the
       new, non-leaking protocol: ≥2 blinded annotators, inter-annotator κ gate,
       adjudication, then gold-vs-auto κ. Rendered by ``inter_annotator``.
    2. **Legacy single-rater** (collapsed expander) — the pre-revision CSV view.
       Kept only for traceability; it is NOT the headline validation because it
       was not blinded.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.tabs.inter_annotator import render_blinded_block


def render(available):
    st.markdown("## Human Annotation")
    st.markdown(
        "*Validation of auto-labels against human judgment. The blinded multi-rater "
        "view is authoritative; the legacy single-rater view is shown only for "
        "traceability to the pre-revision protocol.*"
    )

    # ── 1) Blinded multi-rater block (authoritative) ──────────
    render_blinded_block(st)

    st.markdown("---")

    # ── 2) Legacy single-rater (traceability only) ────────────
    with st.expander(
        "Legacy single-rater view (pre-revision, for traceability only)",
        expanded=False,
    ):
        _render_legacy()


def _render_legacy():
    """Render the pre-revision single-rater comparison from the CSV file."""
    st.caption(
        "⚠️ This view predates the blinded protocol: the annotator saw the auto-label "
        "and similarity fields, so it is not used for headline claims."
    )

    ann_path = Path("results/human_annotation_30.csv")
    if not ann_path.exists():
        st.info("No human annotation file found.")
        return

    ann_df = pd.read_csv(ann_path, comment="#")
    ann_records = ann_df.to_dict("records")
    ann_records = [
        r
        for r in ann_records
        if str(r.get("id", "")).strip() and not str(r.get("id", "")).startswith("#")
    ]

    filled = [r for r in ann_records if r.get("human_label", "") not in (None, "")]
    st.markdown(f"**Annotated:** {len(filled)}/{len(ann_records)} samples")

    if not filled:
        st.info("No annotations found.")
        return

    for rtype in ["safety", "truthfulness", "consistency"]:
        subset = [r for r in filled if r.get("type") == rtype]
        if not subset:
            continue
        st.markdown(f"### {rtype.capitalize()}")

        if rtype == "consistency":
            auto_cons = sum(1 for r in subset if r.get("auto_label") == "consistent")
            human_cons = sum(1 for r in subset if r.get("human_label") == "consistent")
            agree = sum(1 for r in subset if r.get("human_label") == r.get("auto_label"))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Auto: Consistent", f"{auto_cons}/{len(subset)}")
            with c2:
                st.metric("Human: Consistent", f"{human_cons}/{len(subset)}")
            with c3:
                st.metric("Agreement", f"{agree}/{len(subset)} ({agree / len(subset):.0%})")
        else:
            agree = 0
            unverified = 0
            for r in subset:
                hl = r.get("human_label", "")
                if r.get("auto_label") == "unverified":
                    unverified += 1
                    continue
                auto_ok = r.get("auto_correct", "") == "YES"
                human_ok = hl == "correct"
                if auto_ok == human_ok:
                    agree += 1

            total_comp = len(subset) - unverified
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(subset))
            with col2:
                st.metric("Unverified (benign)", unverified)
            with col3:
                st.metric("Agreement", f"{agree}/{total_comp}" if total_comp else "N/A")
            with col4:
                pct = f"{agree / total_comp:.0%}" if total_comp else "N/A"
                st.metric("Agreement %", pct)

            disagree_list = [
                r
                for r in subset
                if r.get("auto_label") != "unverified"
                and (r.get("auto_correct", "") == "YES") != (r.get("human_label", "") == "correct")
            ]
            if disagree_list:
                st.markdown("#### Disagreements")
                for r in disagree_list:
                    with st.expander(f"❌ {r.get('id', '?')}"):
                        st.markdown(f"**Prompt:** {r.get('prompt_text', '')[:150]}")
                        auto_str = "correct" if r.get("auto_correct") == "YES" else "incorrect"
                        st.markdown(
                            f"**Auto:** `{auto_str}` → **Human:** `{r.get('human_label', '')}`"
                        )
                        st.markdown("**Response:**")
                        st.text(r.get("response", "")[:400])
            st.markdown("---")

    # Overall
    st.markdown("### Overall Agreement")
    s_t = [r for r in filled if r.get("type") in ("safety", "truthfulness")]
    comp = sum(1 for r in s_t if r.get("auto_label") != "unverified")
    agree_t = sum(
        1
        for r in s_t
        if r.get("auto_label") != "unverified"
        and (r.get("auto_correct", "") == "YES") == (r.get("human_label", "") == "correct")
    )
    st.success(
        f"**Auto-Human Agreement: {agree_t}/{comp} ({agree_t / comp:.0%})**"
        if comp > 0
        else "**No comparable pairs**"
    )

    st.info(
        "**Interpretation (legacy):** The pre-revision run reported 89% agreement for "
        "safety + truthfulness. Retained only for comparison; it does not reflect the "
        "blinded protocol and is not used for headline claims."
    )

    # Confusion matrix
    st.markdown("### Confusion: Auto vs Human (legacy)")
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in s_t:
        if r.get("auto_label") == "unverified":
            continue
        a = r.get("auto_correct", "") == "YES"
        h = r.get("human_label", "") == "correct"
        if a and h:
            cm["TP"] += 1
        elif a and not h:
            cm["FP"] += 1
        elif not a and h:
            cm["FN"] += 1
        else:
            cm["TN"] += 1
    cm_df = pd.DataFrame(
        [
            ["", "Human: Correct", "Human: Incorrect"],
            ["Auto: Correct", str(cm["TP"]), str(cm["FP"])],
            ["Auto: Incorrect", str(cm["FN"]), str(cm["TN"])],
        ]
    ).astype(str)
    st.dataframe(cm_df, width=400, hide_index=True)
