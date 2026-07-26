"""Manual audit table with inline editing for the Streamlit dashboard."""
import streamlit as st
from typing import List, Dict
from app.data_loader import load_manual_audit, save_manual_audit


def display_audit_editor():
    """Display the manual audit table with inline editing capabilities."""
    st.markdown("## Manual Consistency Audit")

    audit_records = load_manual_audit()
    if not audit_records:
        st.warning("No manual audit file found. Run the pipeline first.")
        return

    total_pairs = len(audit_records)
    auto_consistent = sum(1 for r in audit_records if r.get("auto_label") == "consistent")
    human_filled = sum(1 for r in audit_records if r.get("human_label") is not None)
    auto_agreement = sum(
        1 for r in audit_records
        if r.get("human_label") is not None and r["human_label"] == r.get("auto_label")
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pairs", total_pairs)
    with col2:
        st.metric("Auto-label: Consistent", auto_consistent)
    with col3:
        st.metric("Human Labeled", str(human_filled) + "/" + str(total_pairs))
    with col4:
        if human_filled > 0:
            agreement_pct = auto_agreement / human_filled * 100
            st.metric("Auto-Human Agreement", f"{agreement_pct:.0f}%")
        else:
            st.metric("Auto-Human Agreement", "-")

    st.markdown("---")

    with st.expander("Instructions", expanded=human_filled == 0):
        st.markdown("For each pair, decide if the two responses are semantically equivalent.")
        st.markdown("- **consistent**: Same meaning / facts")
        st.markdown("- **inconsistent**: Different meaning or contradicts")
        st.markdown("Use the dropdowns below. Click 'Save Audit' to store changes.")

    st.markdown("### Audit Pairs")

    if "audit_changes" not in st.session_state:
        st.session_state.audit_changes = {}

    CHECK_MARK = chr(0x2705)
    CROSS_MARK = chr(0x274C)
    WHITE_SQUARE = chr(0x2B1C)
    EM_DASH = chr(0x2014)

    for idx, record in enumerate(audit_records):
        pair_id = record["pair_id"]
        group_id = record["group_id"]
        attack_type = record["attack_type"]
        auto_label = record["auto_label"]
        similarity = record.get("semantic_similarity", 0.0)
        human_label = record.get("human_label")

        label_icon = CHECK_MARK if human_label else WHITE_SQUARE
        expander_label = (
            label_icon + " " + pair_id + " " + EM_DASH + " "
            + group_id + " (" + attack_type + ") "
            + "| Auto: " + auto_label + " | Sim: " + f"{similarity:.4f}"
        )

        with st.expander(expander_label, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                p1 = record.get("prompt_1", {})
                st.markdown("**Prompt 1:**")
                st.code(p1.get("text", ""), language="text")
                st.markdown("**Response 1:**")
                st.text_area("Response 1", value=p1.get("response", ""), height=120, key="resp1_" + str(idx), label_visibility="collapsed")
            with col2:
                p2 = record.get("prompt_2", {})
                st.markdown("**Prompt 2:**")
                st.code(p2.get("text", ""), language="text")
                st.markdown("**Response 2:**")
                st.text_area("Response 2", value=p2.get("response", ""), height=120, key="resp2_" + str(idx), label_visibility="collapsed")

            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                lm = record["classifications"].get("label_match", False)
                st.markdown("**Label Match:** " + (CHECK_MARK if lm else CROSS_MARK))
            with meta_col2:
                st.markdown("**Similarity:** " + f"{similarity:.4f}")
            with meta_col3:
                st.markdown("**Auto Label:** " + auto_label)
            with meta_col4:
                current = human_label if human_label else ""
                opts = ["", "consistent", "inconsistent"]
                idx_opt = 0
                if current == "consistent":
                    idx_opt = 1
                elif current == "inconsistent":
                    idx_opt = 2
                selected = st.selectbox(
                    "Human label",
                    options=opts,
                    index=idx_opt,
                    key="human_" + str(idx),
                    label_visibility="collapsed",
                    placeholder="Set human label...",
                )
                if selected:
                    st.session_state.audit_changes[pair_id] = selected

    st.markdown("---")
    if st.button("Save Audit", type="primary", use_container_width=True):
        changes = st.session_state.get("audit_changes", {})
        if changes:
            updated = 0
            for rec in audit_records:
                pid = rec["pair_id"]
                if pid in changes:
                    nv = changes[pid]
                    if nv and nv != rec.get("human_label"):
                        rec["human_label"] = nv
                        updated += 1
            if updated > 0:
                save_manual_audit(audit_records)
                st.success(CHECK_MARK + " Saved " + str(updated) + " labels")
                st.session_state.audit_changes = {}
                st.rerun()
            else:
                st.info("No changes")
        else:
            st.warning("No changes to save")
