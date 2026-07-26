"""Safety confusion matrix display for the Streamlit dashboard."""
import pandas as pd
import streamlit as st


def display_confusion_matrix(cm, model_label="Model"):
    matrix_data = {
        "Cell": [
            "Malicious Refused",
            "Malicious Complied (Under-refusal)",
            "Benign Answered",
            "Benign Refused (Over-refusal)",
        ],
        "Count": [
            cm.get("malicious_refused", 0),
            cm.get("malicious_complied", 0),
            cm.get("benign_answered", 0),
            cm.get("benign_refused", 0),
        ],
        "Description": [
            "Model correctly refused an attack",
            "Model complied with a malicious prompt",
            "Model correctly answered a benign prompt",
            "Model refused a benign prompt",
        ],
    }
    df = pd.DataFrame(matrix_data)
    st.markdown(f"### Safety Confusion Matrix \u2014 {model_label}")
    st.dataframe(
        df,
        column_config={
            "Cell": st.column_config.Column("Cell", width="medium"),
            "Count": st.column_config.NumberColumn("Count", width="small", format="%d"),
            "Description": st.column_config.Column("Description", width="large"),
        },
        hide_index=True,
        width="stretch",
    )

    malicious_total = cm.get("malicious_refused", 0) + cm.get("malicious_complied", 0)
    benign_total = cm.get("benign_answered", 0) + cm.get("benign_refused", 0)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Attack Refusal Rate", f"{cm.get('malicious_refused',0)/max(malicious_total,1):.0%}")
    with col2:
        st.metric("Under-refusal Rate", f"{cm.get('malicious_complied',0)/max(malicious_total,1):.0%}")
    with col3:
        st.metric("Benign Answered Rate", f"{cm.get('benign_answered',0)/max(benign_total,1):.0%}")
    with col4:
        st.metric("Over-refusal Rate", f"{cm.get('benign_refused',0)/max(benign_total,1):.0%}")


def display_confusion_matrix_comparison(cm_gemma, cm_llama):
    comparison_data = {
        "Cell": ["Malicious Refused", "Malicious Complied", "Benign Answered", "Benign Refused"],
        "Gemma 3 4B": [
            cm_gemma.get(k, 0)
            for k in ["malicious_refused", "malicious_complied", "benign_answered", "benign_refused"]
        ],
        "Llama 3.1 8B": [
            cm_llama.get(k, 0)
            for k in ["malicious_refused", "malicious_complied", "benign_answered", "benign_refused"]
        ],
    }
    df = pd.DataFrame(comparison_data)
    st.markdown("### Safety Confusion Matrix \u2014 Comparison")
    st.dataframe(df, hide_index=True, width="stretch")
