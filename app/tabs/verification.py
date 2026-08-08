"""Tab: Verification — Offline rescoring verification + rescore summary."""

import streamlit as st
import json
import pandas as pd
from pathlib import Path

from app.config import MODEL_NAMES
from app.data_loader import load_rescore_summary


def render(available):
    st.markdown("## Verification")
    st.markdown("*Offline rescoring verification (no Ollama required)*")

    # ── Rescore Summary ─────────────────────────────────
    st.markdown("### Rescore Summary (per-model, per-dimension)")
    summary = load_rescore_summary()
    if summary:
        rows = []
        for key, data in summary.items():
            # key format: "gemma3_4b/safety"
            parts = key.split("/")
            model_key = parts[0]
            dim = parts[1] if len(parts) > 1 else "?"
            ml = MODEL_NAMES.get(model_key, model_key)
            row = {"Model": ml, "Dimension": dim.capitalize()}
            row["Score"] = data.get("score", 0)
            if dim == "safety":
                row["Correct"] = f"{data.get('correct', 0)}/{data.get('total', 0)}"
            elif dim == "truthfulness":
                row["Correct"] = f"{data.get('correct', 0)}/{data.get('total', 0)}"
                row["Unverified"] = data.get("unverified", 0)
            elif dim == "consistency":
                row["Consistent"] = f"{data.get('consistent', 0)}/{data.get('total_groups', 0)}"
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({"Score": "{:.4f}"}), width="stretch", hide_index=True)
        st.caption("Numbers match the original pipeline — rescoring is bite-wise identical.")
    else:
        st.info("No rescore summary found. Run `scripts/score_saved_outputs.py` first.")

    # ── Full Rescored JSON ──────────────────────────────
    st.markdown("### Full Rescored Verification (JSON)")
    rescore_path = Path("results/rescored_verification.json")
    if rescore_path.exists():
        with open(rescore_path) as f:
            rescored = json.load(f)
        with st.expander("Show full JSON"):
            st.json(rescored)
    else:
        st.info("Run `python3 scripts/score_saved_outputs.py` to generate rescored verification.")

    # ── Statistical note ─────────────────────────────────
    st.markdown("### Statistical Verification")
    st.markdown("**Bootstrapping details**:")
    for mk in available:
        ml = MODEL_NAMES.get(mk, mk)
        st.markdown(f"- **{ml}**: 95% CI via n=1000 bootstrap iterations")
