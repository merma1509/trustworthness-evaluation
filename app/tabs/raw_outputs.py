"""Tab 10: Raw Outputs — Individual prompt/response inspection."""

import streamlit as st

from app.config import DIMENSIONS, MODEL_NAMES
from app.data_loader import load_raw_outputs


def render(available):
    st.markdown("## Raw Outputs")
    st.markdown("*Inspect individual prompts and model responses*")

    col1, col2 = st.columns(2)
    with col1:
        model_key = st.selectbox(
            "Model",
            available,
            format_func=lambda x: MODEL_NAMES.get(x, x),
            key="raw_model",
        )
    with col2:
        selected_dim = st.selectbox(
            "Dimension",
            DIMENSIONS,
            key="raw_dim",
        )

    model_label = MODEL_NAMES.get(model_key, model_key)
    st.markdown(f"### {model_label} — {selected_dim.capitalize()}")

    records = load_raw_outputs(model_key, selected_dim)
    if not records:
        st.info(f"No raw outputs found for {model_label} / {selected_dim}")
        return

    st.markdown(f"**Total records:** {len(records)}")
    for i, rec in enumerate(records):
        pid = rec.get("prompt_id", f"#{i}")
        attack_type = rec.get("attack_type", "?")
        is_correct = rec.get("is_correct", False)
        icon = "✅" if is_correct else "❌"
        with st.expander(f"{icon} {pid} ({attack_type})"):
            st.markdown(f"**Prompt:** {rec.get('prompt_text', '')}")
            st.markdown(f"**Expected:** {rec.get('expected_behavior', '')}")
            st.markdown(f"**Actual:** {rec.get('actual_behavior', '')}")
            st.markdown("**Response:**")
            st.text(rec.get("response", "")[:500] or "(empty response)")
            if "provenance" in rec:
                with st.expander("Provenance metadata"):
                    st.json(rec["provenance"])
