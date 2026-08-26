"""Tab 4: Weight Sensitivity — ranking changes across weight configs."""

import pandas as pd
import streamlit as st

from app.components.charts import create_weight_sensitivity_plot
from app.config import MODEL_NAMES, WEIGHT_CONFIGS
from app.data_loader import compute_trustscore, get_dimension_score


def render(available: list, all_scores: dict, all_weight_sensitivity: dict):
    st.markdown("## Weight Sensitivity Analysis")
    st.markdown(
        "TrustScore depends on the chosen weights. Below we show how the ranking changes across different weight configurations."
    )

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

    # Detect ranking flips
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
            + "**Do NOT claim a single winner** — the ranking depends on the weights."
        )
