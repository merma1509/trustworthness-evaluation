"""Tab 1: Overview — Model comparison, TrustScore cards, ranking warnings."""

import pandas as pd
import streamlit as st

from app.components.metrics import display_pipeline_info, display_score_card
from app.config import DIMENSION_LABELS, DIMENSIONS, MODEL_NAMES
from app.data_loader import get_dimension_score


def render(
    available: list,
    all_scores: dict,
    gemma_scores: dict,
    llama_scores: dict,
    gemma_cis: dict,
    llama_cis: dict,
):
    st.markdown("## Overview")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        display_pipeline_info()

    st.markdown("### Model Comparison")

    for model_key in available:
        scores = all_scores.get(model_key, {})
        model_label = MODEL_NAMES.get(model_key, model_key)
        st.markdown(f"#### {model_label}")
        cols = st.columns(4)
        for i, dim in enumerate(DIMENSIONS):
            with cols[i]:
                score = get_dimension_score(scores, dim)
                display_score_card(
                    DIMENSION_LABELS.get(dim, dim),
                    score,
                    color=["#3498db", "#e74c3c", "#2ecc71"][i],
                )
        with cols[3]:
            trust = scores.get("trustworthiness_score", 0)
            display_score_card("TrustScore", trust, color="#9b59b6")
        st.markdown("---")

    # Summary table
    st.markdown("### Final Results")
    results_data = []
    for model_key in available:
        scores = all_scores.get(model_key, {})
        dim_data = scores.get("dimension_scores", {})
        row = {"Model": MODEL_NAMES.get(model_key, model_key)}
        for dim in DIMENSIONS:
            row[dim.capitalize()] = dim_data.get(dim, {}).get("score", 0)
        row["TrustScore"] = scores.get("trustworthiness_score", 0)
        results_data.append(row)
    df = pd.DataFrame(results_data).set_index("Model")
    st.dataframe(df.style.format("{:.4f}"), width="stretch")

    # Ranking warning
    gemma_dims = {d: get_dimension_score(gemma_scores, d) for d in DIMENSIONS}
    llama_dims = {d: get_dimension_score(llama_scores, d) for d in DIMENSIONS}
    dim_winners = []
    for d in DIMENSIONS:
        gs = gemma_dims[d]
        ls = llama_dims[d]
        if gs > ls:
            dim_winners.append(f"Gemma wins on **{d.capitalize()}** ({gs:.4f} vs {ls:.4f})")
        elif ls > gs:
            dim_winners.append(f"Llama wins on **{d.capitalize()}** ({ls:.4f} vs {gs:.4f})")
        else:
            dim_winners.append(f"**{d.capitalize()}** tie ({gs:.4f})")

    instab_msg = "\n".join(["- " + w for w in dim_winners])
    instab_msg += "\n- The overall ranking **depends on the chosen weights**"
    instab_msg += "\n- All confidence intervals **overlap** — differences are not statistically significant"
    warning_icon = chr(0x26A0) + chr(0xFE0F)
    st.warning(f"{warning_icon} **Ranking Instability Warning**\n\n{instab_msg}")
