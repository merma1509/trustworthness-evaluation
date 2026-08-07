"""Tab 9: Verification — Offline rescoring verification."""

import streamlit as st
import json
from pathlib import Path
from app.config import MODEL_NAMES


def render(available):
    st.markdown("## Verification")
    st.markdown("*Offline rescoring verification (no Ollama required)*")

    st.markdown("### Rescored Verification")
    rescore_path = Path("results/rescored_verification.json")
    if rescore_path.exists():
        with open(rescore_path) as f:
            rescored = json.load(f)
        st.json(rescored)
    else:
        st.info("Run `python3 scripts/score_saved_outputs.py` to generate rescored verification.")

    st.markdown("### Statistical Verification")
    st.markdown("**Bootstrapping details**:")
    for mk in available:
        ml = MODEL_NAMES.get(mk, mk)
        st.markdown(f"- **{ml}**: 95% CI via n=1000 bootstrap iterations")
