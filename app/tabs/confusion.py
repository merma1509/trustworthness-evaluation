"""Tab 3: Confusion Matrix — Safety confusion matrices + comparison."""

import streamlit as st

from app.components.charts import create_confusion_matrix_heatmap
from app.components.confusion_matrix import (
    display_confusion_matrix,
    display_confusion_matrix_comparison,
)


def render(available: list, gemma_cm: dict, llama_cm: dict, gemma_scores: dict, llama_scores: dict):
    st.markdown("## Safety Confusion Matrix")

    col1, col2 = st.columns(2)
    with col1:
        display_confusion_matrix(gemma_cm, "Gemma 3 4B")
    with col2:
        display_confusion_matrix(llama_cm, "Llama 3.1 8B")

    st.markdown("### Comparison")
    fig_cm = create_confusion_matrix_heatmap(gemma_cm, llama_cm)
    st.plotly_chart(fig_cm, width="stretch")
    display_confusion_matrix_comparison(gemma_cm, llama_cm)

    # Build key findings dynamically from confusion matrix
    gemma_mal_ref = gemma_cm.get("malicious_refused", 0)
    gemma_mal_comp = gemma_cm.get("malicious_complied", 0)
    gemma_ben_ref = gemma_cm.get("benign_refused", 0)
    gemma_mal_total = gemma_mal_ref + gemma_mal_comp
    llama_mal_ref = llama_cm.get("malicious_refused", 0)
    llama_mal_comp = llama_cm.get("malicious_complied", 0)
    llama_ben_ref = llama_cm.get("benign_refused", 0)
    if gemma_mal_ref >= llama_mal_ref:
        ref_user, ref_best, ref_other, ref_worst = "Gemma", gemma_mal_ref, "Llama", llama_mal_ref
    else:
        ref_user, ref_best, ref_other, ref_worst = "Llama", llama_mal_ref, "Gemma", gemma_mal_ref

    ref_total = gemma_mal_total  # same for both models
    und_total = gemma_mal_total

    if gemma_mal_comp <= llama_mal_comp:
        und_user, und_best, und_other, und_worst = "Gemma", gemma_mal_comp, "Llama", llama_mal_comp
    else:
        und_user, und_best, und_other, und_worst = "Llama", llama_mal_comp, "Gemma", gemma_mal_comp

    if gemma_ben_ref <= llama_ben_ref:
        ovr_user, ovr_best, ovr_other, ovr_worst = "Gemma", gemma_ben_ref, "Llama", llama_ben_ref
    else:
        ovr_user, ovr_best, ovr_other, ovr_worst = "Llama", llama_ben_ref, "Gemma", gemma_ben_ref

    st.info(
        f"**Key findings:**\n\n"
        f"- **{ref_user}** refuses more malicious prompts: "
        f"**{ref_best}/{ref_total}** vs {ref_other}'s **{ref_worst}/{ref_total}**\n"
        f"- **{ovr_user}** has fewer over-refusal cases: "
        f"**{ovr_best}** vs {ovr_other}'s **{ovr_worst}**\n"
        f"- **{und_user}** has lower under-refusal: "
        f"**{und_best}/{und_total}** vs {und_other}'s **{und_worst}/{und_total}**"
    )

