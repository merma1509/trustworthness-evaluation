"""Tab 2: Dimensions — Bar chart, confidence intervals, consistency details."""

import streamlit as st
import pandas as pd

from app.config import DIMENSIONS, MODEL_NAMES
from app.components.charts import create_dimension_bar_chart
from app.data_loader import get_dimension_score, load_scores


def render(
    available: list,
    all_scores: dict,
    gemma_scores: dict,
    llama_scores: dict,
    gemma_cis: dict,
    llama_cis: dict,
    all_cis: dict,
):
    st.markdown("## Dimension Scores")

    fig = create_dimension_bar_chart(gemma_scores, llama_scores, gemma_cis, llama_cis)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Confidence Intervals (95%, Bootstrap n=1000)")
    ci_data = []
    for model_key, cis in all_cis.items():
        for dim in DIMENSIONS:
            ci = cis.get(dim, {})
            ci_data.append(
                {
                    "Model": MODEL_NAMES.get(model_key, model_key),
                    "Dimension": dim.capitalize(),
                    "Score": get_dimension_score(all_scores.get(model_key, {}), dim),
                    "CI Lower": ci.get("ci_lower", 0),
                    "CI Upper": ci.get("ci_upper", 0),
                    "Width": ci.get("ci_upper", 0) - ci.get("ci_lower", 0),
                }
            )
    ci_df = pd.DataFrame(ci_data)
    st.dataframe(
        ci_df.style.format(
            {"Score": "{:.4f}", "CI Lower": "{:.4f}", "CI Upper": "{:.4f}", "Width": "{:.4f}"}
        ),
        width="stretch",
    )
    st.caption("All confidence intervals overlap — rankings are not statistically significant")

    st.markdown("### Consistency Group Details")
    for model_key in available:
        model_label = MODEL_NAMES.get(model_key, model_key)
        cons = load_scores(model_key)
        if cons:
            ds = cons.get("dimension_scores", {}).get("consistency", {})
            st.metric(
                f"{model_label}: Consistent Groups",
                f"{ds.get('consistent_groups', '?')}/{ds.get('total_groups', '?')}",
            )
