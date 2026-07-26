"""Metric cards and score displays for the Streamlit dashboard."""
import streamlit as st
from app.config import DIMENSION_LABELS, DIMENSION_COLORS, interpret_score


def display_score_card(label, score, color="#3498db", help_text=""):
    trust_level = interpret_score(score)
    st.markdown(
        f"""<div style="background-color:{color}15;border:1px solid {color}40;border-radius:10px;padding:15px;margin:5px 0;text-align:center;">
        <div style="font-size:14px;color:#555;margin-bottom:5px;">{label}</div>
        <div style="font-size:36px;font-weight:bold;color:{color};">{score:.4f}</div>
        <div style="font-size:13px;color:#888;margin-top:5px;">{trust_level}</div></div>""",
        unsafe_allow_html=True,
    )
    if help_text:
        st.caption(help_text)


def display_pipeline_info():
    st.markdown("### Pipeline Info")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Models Evaluated", "2")
    with info_cols[1]:
        st.metric("Total Prompts", "105")
    with info_cols[2]:
        st.metric("Dimensions", "3")
