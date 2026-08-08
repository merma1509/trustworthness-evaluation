"""Tab 5: Manual Audit — Consistency pair editor for human labeling."""

import streamlit as st
from app.components.manual_audit import display_audit_editor


def render(available):
    st.markdown("## Manual Audit")
    st.markdown("*Edit consistency audit labels*")
    display_audit_editor()
