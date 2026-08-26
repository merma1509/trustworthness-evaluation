"""Tab 2: Dimensions — Bar chart, confidence intervals, consistency details."""

import pandas as pd
import streamlit as st

from app.components.charts import create_dimension_bar_chart
from app.config import DIMENSIONS, MODEL_NAMES
from app.data_loader import get_dimension_score, load_paired_comparison, load_scores


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
    st.caption(
        "Marginal 95% CIs above are for illustration. Whether a "
        "model difference is **significant** is judged by the paired "
        "(safety/truthfulness) or clustered (consistency) design below — "
        "not by marginal-CI overlap."
    )

    st.markdown("### Significance of Model Differences (Paired / Clustered)")
    with st.container(border=True):
        st.markdown(
            "Because the two models answer the **same prompts**, their outputs "
            "are correlated — overlapping marginal CIs are a statistically "
            "invalid way to compare them. Significance is instead based on the "
            "**paired bootstrap** (safety/truthfulness) and the **clustered "
            "bootstrap** (consistency, resampling whole groups)."
        )
        paired = load_paired_comparison()
        if paired and paired.get("dimensions"):
            rows = []
            for dim in DIMENSIONS:
                d = paired["dimensions"].get(dim, {})
                if not d:
                    continue
                diff = d.get("mean_difference", 0)
                lo, hi = d.get("ci_lower", 0), d.get("ci_upper", 0)
                p = d.get("p_value")
                significant = not (lo <= 0 <= hi)
                n = d.get("n_pairs") or d.get("n_groups", "?")
                rows.append(
                    {
                        "Dimension": dim.capitalize(),
                        "Mean Difference": diff,
                        "Paired CI Lower": lo,
                        "Paired CI Upper": hi,
                        "p-value": p if p is not None else "n/a",
                        "Significant (5%)": "Yes" if significant else "No",
                        "n": n,
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {
                        "Mean Difference": "{:+.4f}",
                        "Paired CI Lower": "{:.4f}",
                        "Paired CI Upper": "{:.4f}",
                        "p-value": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            any_sig = any(
                not (d.get("ci_lower", 0) <= 0 <= d.get("ci_upper", 0))
                for d in paired.get("dimensions", {}).values()
            )
            if any_sig:
                st.success("A statistically significant difference exists (CI excludes 0).")
            else:
                st.info(
                    "No paired difference is statistically significant at 5% "
                    "(all paired CIs include 0) — conclusions should stay "
                    "qualitative / weight-dependent."
                )
        else:
            st.info(
                "Paired-comparison data unavailable. Run `make run` so "
                "`results/paired_comparison.json` is generated."
            )

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
