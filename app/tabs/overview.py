"""Tab 1: Overview — Model comparison, TrustScore cards, ranking warnings."""

import pandas as pd
import streamlit as st

from app.components.metrics import display_pipeline_info, display_score_card
from app.config import DIMENSION_LABELS, DIMENSIONS, MODEL_NAMES, RESULTS_DIR
from app.data_loader import get_dimension_score, load_paired_comparison


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

    # Fig 1 — measurement-validation loop (text can be locally supplied).
    with st.expander("Instrument pipeline loop (Fig 1)", expanded=False):
        loop_path = RESULTS_DIR / "pipeline_loop.png"
        if loop_path.exists():
            st.image(str(loop_path), caption="Measurement-validation loop")
        else:
            st.caption("Generate with: `make pipeline-figure`")
        st.markdown(
            "**How to read it.** The auto-scorer runs on every prompt; a "
            "calibration/held-out split feeds a per-dimension reliability "
            "estimate; a **κ gate** decides whether to trust the scorer directly "
            "or route samples to blinded human re-annotation. The loop is the "
            "durable contribution — no run-specific numbers live in the diagram."
        )

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

    # Statistically-significant differences come from the PAIRED design
    # (safety/truthfulness) and the CLUSTERED design (consistency) — NOT from
    # eyeballing overlapping marginal CIs, which is statistically invalid for
    # correlated model outputs.
    paired = load_paired_comparison()
    sig_lines = []
    if paired and paired.get("dimensions"):
        for dim in DIMENSIONS:
            d = paired["dimensions"].get(dim, {})
            if not d:
                continue
            diff = d.get("mean_difference", 0)
            lo, hi = d.get("ci_lower", 0), d.get("ci_upper", 0)
            # CI excludes zero -> significant at 5% (two-sided).
            significant = not (lo <= 0 <= hi)
            p = d.get("p_value")
            p_str = f", p={p:.3f}" if p is not None else ""
            n = d.get("n_pairs") or d.get("n_groups", "?")
            verdict = "significant" if significant else "not significant"
            sig_lines.append(
                f"- **{dim.capitalize()}**: Δ={diff:+.4f} "
                f"[{lo:.4f}, {hi:.4f}]{p_str} → {verdict} (paired n={n})"
            )
    else:
        sig_lines.append("- Paired-comparison data unavailable (run `make run` to generate it).")
    instab_msg += "\n\n**Paired / clustered significance (not CI-overlap):**\n" + "\n".join(
        sig_lines
    )

    warning_icon = chr(0x26A0) + chr(0xFE0F)
    st.warning(f"{warning_icon} **Ranking Instability Warning**\n\n{instab_msg}")
