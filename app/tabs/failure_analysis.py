"""Tab 7: Failure Analysis — Per-attack-type breakdown for each dimension."""

import pandas as pd
import streamlit as st

from app.config import DIMENSIONS, MODEL_NAMES
from app.data_loader import load_raw_outputs


def render(available):
    st.markdown("## Failure Analysis")
    st.markdown("*Where does each model fail? Per-attack-type breakdown.*")

    raw_data = {}
    for dim in DIMENSIONS:
        for mk in available:
            raw_data[(mk, dim)] = load_raw_outputs(mk, dim)

    for dim in DIMENSIONS:
        st.markdown(f"### {dim.capitalize()}")
        rows = []
        for mk in available:
            records = raw_data.get((mk, dim), [])
            atypes = set(r.get("attack_type", "unknown") for r in records)
            for at in sorted(atypes):
                subset = [r for r in records if r.get("attack_type") == at]
                if not subset:
                    continue
                correct = sum(1 for r in subset if r.get("is_correct", False))
                total = len(subset)
                rows.append({
                    "Model": MODEL_NAMES.get(mk, mk),
                    "Attack Type": at,
                    "Correct": f"{correct}/{total}",
                    "Accuracy": correct / total if total else 0,
                    "Total": total,
                })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({"Accuracy": "{:.0%}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(f"No {dim} data available.")

        st.markdown("---")
