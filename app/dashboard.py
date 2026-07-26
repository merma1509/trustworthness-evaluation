#!/usr/bin/env python3
"""Trustworthiness Evaluation — Streamlit Dashboard

Usage:
    streamlit run app/dashboard.py
    make dashboard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Page config MUST be first
st.set_page_config(
    page_title="Trustworthiness Evaluation",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.charts import (  # noqa: E402
    create_confusion_matrix_heatmap,
    create_dimension_bar_chart,
    create_weight_sensitivity_plot,
)
from app.components.confusion_matrix import (  # noqa: E402
    display_confusion_matrix,
    display_confusion_matrix_comparison,
)
from app.components.manual_audit import display_audit_editor  # noqa: E402
from app.components.metrics import display_pipeline_info, display_score_card  # noqa: E402
from app.config import DIMENSION_LABELS, DIMENSIONS, MODEL_NAMES, WEIGHT_CONFIGS  # noqa: E402
from app.data_loader import (  # noqa: E402
    compute_trustscore,
    get_confusion_matrix,
    get_dimension_score,
    load_all_confidence_intervals,
    load_all_scores,
    load_all_weight_sensitivity,
    load_scores,
    models_available,
)

# ─── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Trustworthiness Eval")
st.sidebar.markdown("---")

available = models_available()
if not available:
    st.sidebar.error("No results found. Run the pipeline first:\n\n`./demo.sh`")
else:
    st.sidebar.success(f"{len(available)} model(s) available")

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available,
        format_func=lambda k: MODEL_NAMES.get(k, k),
        index=0,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown("[GitHub Repo](https://github.com/merma1509/trustworthness-evaluation)")
st.sidebar.markdown("[README](https://github.com/merma1509/trustworthness-evaluation#readme)")

# Info about pipeline
st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: `./demo.sh` | Models: Gemma 3 4B, Llama 3.1 8B | Prompts: 105")


# ─── Main Content ───────────────────────────────────────────
st.title("Trustworthiness Evaluation")
st.markdown("*A Lightweight Validation Study of Open-Source LLMs*")

if not available:
    st.warning(
        """### No results found\n\nRun the full pipeline first:\n```bash\n./demo.sh\n```\nThen refresh this dashboard."""
    )
    st.stop()

# Load all data
all_scores = load_all_scores()
all_cis = load_all_confidence_intervals()
all_weight_sensitivity = load_all_weight_sensitivity()
gemma_scores = all_scores.get("gemma3_4b", {})
llama_scores = all_scores.get("llama3.1_8b", {})
gemma_cis = all_cis.get("gemma3_4b", {})
llama_cis = all_cis.get("llama3.1_8b", {})
gemma_cm = get_confusion_matrix(gemma_scores)
llama_cm = get_confusion_matrix(llama_scores)


# ─── Tabs ───────────────────────────────────────────────────
tab_overview, tab_dimensions, tab_confusion, tab_weights, tab_audit, tab_raw = st.tabs(
    [
        "Overview",
        "Dimensions",
        "Confusion Matrix",
        "Weight Sensitivity",
        "Manual Audit",
        "Raw Outputs",
    ]
)

# ─── TAB 1: Overview ────────────────────────────────────────
with tab_overview:
    st.markdown("## Overview")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        display_pipeline_info()

    st.markdown("### Model Comparison")

    # Metric cards
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
    import pandas as pd

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
    # Build ranking warning dynamically from actual scores
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
    instab_msg += "\n- All confidence intervals **overlap** \u2014 differences are not statistically significant"
    warning_icon = chr(0x26A0) + chr(0xFE0F)
    st.warning(f"{warning_icon} **Ranking Instability Warning**\n\n{instab_msg}")

# ─── TAB 2: Dimensions ──────────────────────────────────────
with tab_dimensions:
    st.markdown("## Dimension Scores")

    # Bar chart
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

    # Per-group detail
    st.markdown("### Consistency Group Details")
    gemma_cons = load_scores("gemma3_4b")
    if gemma_cons:
        ds = gemma_cons.get("dimension_scores", {}).get("consistency", {})
        st.metric(
            "Consistent Groups", f"{ds.get('consistent_groups', '?')}/{ds.get('total_groups', '?')}"
        )

# ─── TAB 3: Confusion Matrix ────────────────────────────────
with tab_confusion:
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
    gemma_ben_ans = gemma_cm.get("benign_answered", 0)
    gemma_ben_ref = gemma_cm.get("benign_refused", 0)
    gemma_mal_total = gemma_mal_ref + gemma_mal_comp
    gemma_ben_total = gemma_ben_ans + gemma_ben_ref

    llama_mal_ref = llama_cm.get("malicious_refused", 0)
    llama_mal_comp = llama_cm.get("malicious_complied", 0)
    llama_ben_ans = llama_cm.get("benign_answered", 0)
    llama_ben_ref = llama_cm.get("benign_refused", 0)
    llama_mal_total = llama_mal_ref + llama_mal_comp
    llama_ben_total = llama_ben_ans + llama_ben_ref

    # Dynamically determine wording based on actual data
    if gemma_mal_ref >= llama_mal_ref:
        ref_user = "Gemma"
        ref_best = gemma_mal_ref
        ref_other = "Llama"
        ref_worst = llama_mal_ref
    else:
        ref_user = "Llama"
        ref_best = llama_mal_ref
        ref_other = "Gemma"
        ref_worst = gemma_mal_ref
    ref_total = gemma_mal_total  # same for both

    if gemma_mal_comp <= llama_mal_comp:
        und_user = "Gemma"
        und_best = gemma_mal_comp
        und_other = "Llama"
        und_worst = llama_mal_comp
    else:
        und_user = "Llama"
        und_best = llama_mal_comp
        und_other = "Gemma"
        und_worst = gemma_mal_comp
    und_total = gemma_mal_total

    if gemma_ben_ref <= llama_ben_ref:
        ovr_user = "Gemma"
        ovr_best = gemma_ben_ref
        ovr_other = "Llama"
        ovr_worst = llama_ben_ref
    else:
        ovr_user = "Llama"
        ovr_best = llama_ben_ref
        ovr_other = "Gemma"
        ovr_worst = gemma_ben_ref

    st.info(
        f"**Key findings:**\n\n"
        f"- **{ref_user}** refuses more malicious prompts: "
        f"**{ref_best}/{ref_total}** vs {ref_other}'s **{ref_worst}/{ref_total}**\n"
        f"- **{ovr_user}** has fewer over-refusal cases: "
        f"**{ovr_best}** vs {ovr_other}'s **{ovr_worst}**\n"
        f"- **{und_user}** has lower under-refusal: "
        f"**{und_best}/{und_total}** vs {und_other}'s **{und_worst}/{und_total}**"
    )

with tab_weights:
    st.markdown("## Weight Sensitivity Analysis")

    st.markdown("""
    TrustScore depends on the chosen weights. Here we show how the ranking
    changes across different weight configurations.
    """)

    fig_ws = create_weight_sensitivity_plot(all_weight_sensitivity)
    st.plotly_chart(fig_ws, width="stretch")

    st.markdown("### Weight Configurations")

    ws_data = []
    for cfg in WEIGHT_CONFIGS:
        row = {"Config": cfg["name"], "w_s": cfg["w_s"], "w_t": cfg["w_t"], "w_c": cfg["w_c"]}
        for model_key in available:
            scores = all_scores.get(model_key, {})
            s = get_dimension_score(scores, "safety")
            t = get_dimension_score(scores, "truthfulness")
            c = get_dimension_score(scores, "consistency")
            row[MODEL_NAMES.get(model_key, model_key)] = compute_trustscore(
                s, t, c, cfg["w_s"], cfg["w_t"], cfg["w_c"]
            )
        ws_data.append(row)

    ws_df = pd.DataFrame(ws_data)
    # Add winner column
    if len(available) >= 2:
        labels = [MODEL_NAMES.get(k, k) for k in available]
        winners = []
        for _, row in ws_df.iterrows():
            if row[labels[0]] > row[labels[1]]:
                winners.append(labels[0])
            elif row[labels[1]] > row[labels[0]]:
                winners.append(labels[1])
            else:
                winners.append("Tie")
        ws_df["Winner"] = winners

    st.dataframe(
        ws_df.style.format({k: "{:.4f}" for k in ws_df.columns if k not in ["Config", "Winner"]}),
        width="stretch",
    )

    # Detect ranking flips dynamically from actual weight sensitivity data
    flip_msgs = []
    if all_weight_sensitivity:
        gemma_ws = all_weight_sensitivity.get("gemma3_4b", [])
        llama_ws = all_weight_sensitivity.get("llama3.1_8b", [])
        for g_rec, l_rec in zip(gemma_ws, llama_ws):
            cfg_name = g_rec.get("name", l_rec.get("name", "?"))
            g_score = g_rec.get("score", 0)
            l_score = l_rec.get("score", 0)
            if abs(g_score - l_score) < 0.0001:
                continue
            if gemma_ws and llama_ws:
                base_g = gemma_ws[0].get("score", 0)
                base_l = llama_ws[0].get("score", 0)
                base_winner = "Gemma" if base_g > base_l else "Llama"
                this_winner = "Gemma" if g_score > l_score else "Llama"
                if base_winner != this_winner:
                    flip_msgs.append(
                        f"Under **{cfg_name}**, **{this_winner}** wins "
                        f"(**{max(g_score, l_score):.4f}** vs **{min(g_score, l_score):.4f}**)"
                    )
    if flip_msgs:
        flip_text = "\n".join(["- " + m for m in flip_msgs])
        flip_icon = chr(0x26A0) + chr(0xFE0F)
        st.warning(
            flip_icon
            + " **Ranking Flip Detected!**\n\n"
            + flip_text
            + "\n"
            + "**Do NOT claim a single winner** "
            + chr(0x2014)
            + " the ranking depends on the weights."
        )
# ─── TAB 5: Manual Audit ────────────────────────────────────
with tab_audit:
    display_audit_editor()

# ─── TAB 6: Raw Outputs ─────────────────────────────────────
with tab_raw:
    st.markdown("## Raw Outputs")

    selected_dim = st.selectbox(
        "Select Dimension", DIMENSIONS, format_func=lambda d: d.capitalize()
    )

    raw_files = {
        "gemma3_4b": load_scores("gemma3_4b"),
        "llama3.1_8b": load_scores("llama3.1_8b"),
    }

    for model_key in available:
        model_label = MODEL_NAMES.get(model_key, model_key)
        st.markdown(f"### {model_label} — {selected_dim.capitalize()}")

        from app.data_loader import load_raw_outputs  # noqa: E402

        records = load_raw_outputs(model_key, selected_dim)
        if not records:
            st.info(f"No raw outputs for {model_label}/{selected_dim}")
            continue

        # Summary
        correct = sum(1 for r in records if r.get("is_correct", False))
        total = len(records)
        st.metric("Score", f"{correct}/{total} ({correct / max(total, 1):.2%})")

        # Show expandable records
        for rec in records[:10]:  # Show first 10
            pid = rec.get("prompt_id", "?")
            att = rec.get("attack_type", "?")
            is_correct = rec.get("is_correct", False)
            icon = "\u2705" if is_correct else "\u274c"
            with st.expander(f"{icon} {pid} ({att})", expanded=False):
                st.markdown("**Prompt:**")
                st.code(rec.get("prompt_text", ""), language="text")
                st.markdown("**Response:**")
                st.text(rec.get("response", "")[:1000])
                if len(rec.get("response", "")) > 1000:
                    st.caption(f"...({len(rec['response'])} total chars)")
                st.markdown(
                    f"**Expected:** {rec.get('expected_behavior', '?')} | "
                    f"**Actual:** {rec.get('actual_behavior', '?')}"
                )


# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Built for the course 'Security and Interpretability of Machine Learning' at Innopolis University. "
    "[GitHub](https://github.com/merma1509/trustworthness-evaluation)*"
)
