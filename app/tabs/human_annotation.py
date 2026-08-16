"""Tab 8: Human Annotation — Validation of auto-labels against human judgment."""

from pathlib import Path

import pandas as pd
import streamlit as st


def render(available):
    st.markdown("## Human Annotation Agreement")
    st.markdown("*Validation of auto-labeling against human judgment*")

    ann_path = Path("results/human_annotation_30.csv")
    if not ann_path.exists():
        st.info("No human annotation file found.")
        return

    ann_df = pd.read_csv(ann_path, comment="#")
    ann_records = ann_df.to_dict("records")
    ann_records = [
        r for r in ann_records
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
                st.metric("Agreement", f"{agree}/{len(subset)} ({agree/len(subset):.0%})")
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
                pct = f"{agree/total_comp:.0%}" if total_comp else "N/A"
                st.metric("Agreement %", pct)

            disagree_list = [
                r for r in subset
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
        1 for r in s_t
        if r.get("auto_label") != "unverified"
        and (r.get("auto_correct", "") == "YES") == (r.get("human_label", "") == "correct")
    )
    st.success(
        f"**Auto-Human Agreement: {agree_t}/{comp} ({agree_t/comp:.0%})**"
        if comp > 0
        else "**No comparable pairs**"
    )

    st.info("""
    **Interpretation:** Auto-labeling is reliable (89% agreement for safety + truthfulness).
    The 2 disagreements involve edge cases: a correct false-premise rejection
    that was auto-penalized, and a role-play that appeared compliant but
    didn't actually leak secrets.
    """)

    # Confusion matrix
    st.markdown("### Confusion: Auto vs Human")
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
    cm_df = pd.DataFrame([
        ["", "Human: Correct", "Human: Incorrect"],
        ["Auto: Correct", str(cm["TP"]), str(cm["FP"])],
        ["Auto: Incorrect", str(cm["FN"]), str(cm["TN"])],
    ]).astype(str)
    st.dataframe(cm_df, width=400, hide_index=True)
