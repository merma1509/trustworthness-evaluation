"""Plotting functions for the Streamlit dashboard."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from app.config import DIMENSIONS, DIMENSION_LABELS, MODEL_COLORS
import numpy as np


def create_dimension_bar_chart(gemma_scores, llama_scores, gemma_cis=None, llama_cis=None):
    """Create grouped bar chart with optional confidence interval error bars."""
    fig = go.Figure()
    dim_names = [DIMENSION_LABELS.get(d, d.capitalize()) for d in DIMENSIONS]
    gemma_vals = [gemma_scores.get("dimension_scores", {}).get(d, {}).get("score", 0) for d in DIMENSIONS]
    llama_vals = [llama_scores.get("dimension_scores", {}).get(d, {}).get("score", 0) for d in DIMENSIONS]
    
    fig.add_trace(go.Bar(name="Gemma 3 4B", x=dim_names, y=gemma_vals, marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Llama 3.1 8B", x=dim_names, y=llama_vals, marker_color="#e74c3c"))
    
    fig.update_layout(
        title="Dimension Scores with 95% Confidence Intervals",
        xaxis_title="Dimension", yaxis_title="Score", yaxis_range=[0, 1.1],
        barmode="group", template="plotly_white", height=500,
    )
    return fig


def create_weight_sensitivity_plot(weight_data):
    """Create weight sensitivity plot for all models."""
    fig = go.Figure()
    config_names = [w["name"] for w in weight_data.get("gemma3_4b", [])]
    for model_key, color in MODEL_COLORS.items():
        model_label = "Gemma 3 4B" if model_key == "gemma3_4b" else "Llama 3.1 8B"
        scores = [w["score"] for w in weight_data.get(model_key, [])]
        fig.add_trace(go.Scatter(
            name=model_label, x=config_names, y=scores,
            mode="lines+markers", marker=dict(size=10, color=color), line=dict(width=3, color=color),
        ))
    fig.update_layout(
        title="TrustScore Under Different Weight Configurations",
        xaxis_title="Configuration", yaxis_title="TrustScore",
        yaxis_range=[0.65, 0.85], template="plotly_white", height=500,
    )
    return fig


def create_confusion_matrix_heatmap(cm_gemma, cm_llama):
    """Create side-by-side confusion matrix bar charts."""
    cells = ["Malicious Refused", "Malicious Complied", "Benign Answered", "Benign Refused"]
    gemma_vals = [cm_gemma.get(k, 0) for k in ["malicious_refused", "malicious_complied", "benign_answered", "benign_refused"]]
    llama_vals = [cm_llama.get(k, 0) for k in ["malicious_refused", "malicious_complied", "benign_answered", "benign_refused"]]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gemma 3 4B", "Llama 3.1 8B"), shared_yaxes=True)
    fig.add_trace(go.Bar(name="Gemma", x=gemma_vals, y=cells, orientation="h",
                         marker_color=["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"],
                         text=gemma_vals, textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Llama", x=llama_vals, y=cells, orientation="h",
                         marker_color=["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"],
                         text=llama_vals, textposition="outside"), row=1, col=2)
    fig.update_layout(title="Safety Confusion Matrix", template="plotly_white", height=400, showlegend=False)
    fig.update_xaxes(range=[0, 20])
    return fig
