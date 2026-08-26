#!/usr/bin/env python3
"""Trustworthiness Evaluation — Streamlit Dashboard (modular)

Slim orchestrator: imports data, creates tabs, delegates rendering.
Each tab lives in app/tabs/<name>.py with a single render() function.
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

from app.config import MODEL_NAMES
from app.data_loader import (
    get_confusion_matrix,
    load_agreement_report,
    load_all_confidence_intervals,
    load_all_scores,
    load_all_weight_sensitivity,
    load_ranking_stability,
    load_rescore_summary,
    models_available,
)
from app.tabs.confusion import render as render_confusion
from app.tabs.dimensions import render as render_dimensions
from app.tabs.failure_analysis import render as render_failure
from app.tabs.human_annotation import render as render_human
from app.tabs.manual_audit_tab import render as render_audit

# ─── Tab modules ──────────────────────────────────────────
from app.tabs.overview import render as render_overview
from app.tabs.raw_outputs import render as render_raw
from app.tabs.research_question import render as render_research
from app.tabs.verification import render as render_verification
from app.tabs.weight_sensitivity import render as render_weights

# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.title("Trustworthiness Eval")
st.sidebar.markdown("---")

available = models_available()
if not available:
    st.sidebar.error("No results found. Run the pipeline first:\n\n`./demo.sh`")
else:
    st.sidebar.success(f"{len(available)} model(s) available")
    st.sidebar.selectbox(
        "Select Model",
        options=available,
        format_func=lambda k: MODEL_NAMES.get(k, k),
        index=0,
        key="sidebar_model",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown("[GitHub Repo](https://github.com/merma1509/trustworthness-evaluation)")
st.sidebar.markdown("[README](https://github.com/merma1509/trustworthness-evaluation#readme)")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Pipeline: `./demo.sh` | Models: {', '.join(MODEL_NAMES.values())} | Prompts: 105"
)

# ─── Main Content ───────────────────────────────────────────
st.title("Trustworthiness Evaluation")
st.markdown(
    "*When can a very small, cheap, local evaluation be trusted, and how do we know when it fails?*"
)

if not available:
    st.warning("\u26a0\ufe0f No evaluation results found. Run `./demo.sh` first.")
    st.stop()

# Load all data once
all_scores = load_all_scores()
all_cis = load_all_confidence_intervals()
all_weight_sensitivity = load_all_weight_sensitivity()
agreement_report = load_agreement_report()
ranking_stability = load_ranking_stability()
rescore_summary = load_rescore_summary()
gemma_scores = all_scores.get("gemma3_4b", {})
llama_scores = all_scores.get("llama3.1_8b", {})
gemma_cis = all_cis.get("gemma3_4b", {})
llama_cis = all_cis.get("llama3.1_8b", {})
gemma_cm = get_confusion_matrix(gemma_scores)
llama_cm = get_confusion_matrix(llama_scores)

# ─── Tabs ────────────────────────────────────────────────────
(
    tab_overview,
    tab_dimensions,
    tab_confusion,
    tab_weights,
    tab_research,
    tab_failure,
    tab_human,
    tab_audit,
    tab_verification,
    tab_raw,
) = st.tabs(
    [
        "Overview",
        "Dimensions",
        "Confusion Matrix",
        "Weight Sensitivity",
        "Research Questions",
        "Failure Analysis",
        "Human Annotation",
        "Manual Audit",
        "Verification",
        "Raw Outputs",
    ]
)

with tab_overview:
    render_overview(available, all_scores, gemma_scores, llama_scores, gemma_cis, llama_cis)

with tab_dimensions:
    render_dimensions(
        available, all_scores, gemma_scores, llama_scores, gemma_cis, llama_cis, all_cis
    )

with tab_confusion:
    render_confusion(available, gemma_cm, llama_cm, gemma_scores, llama_scores)

with tab_weights:
    render_weights(available, all_scores, all_weight_sensitivity)

with tab_research:
    render_research(
        available, gemma_scores, llama_scores, gemma_cis, llama_cis, all_scores, all_cis
    )

with tab_failure:
    render_failure(available)

with tab_human:
    render_human(available)

with tab_audit:
    render_audit(available)

with tab_verification:
    render_verification(available)

with tab_raw:
    render_raw(available)

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Built for the course 'Security and Interpretability of Machine Learning' at Innopolis University. "
    "[GitHub](https://github.com/merma1509/trustworthness-evaluation)*"
)
