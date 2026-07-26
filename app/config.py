"""Configuration constants for the Streamlit dashboard."""
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
RAW_OUTPUTS_DIR = RESULTS_DIR / "raw_outputs"

MODEL_NAMES = {
    "gemma3_4b": "Gemma 3 4B",
    "llama3.1_8b": "Llama 3.1 8B",
}

DIMENSIONS = ["safety", "truthfulness", "consistency"]
DIMENSION_LABELS = {
    "safety": "Safety \U0001f6e1\ufe0f",
    "truthfulness": "Truthfulness \U0001f4d6",
    "consistency": "Consistency \U0001f504",
}
DIMENSION_COLORS = {
    "safety": "#3498db",
    "truthfulness": "#e74c3c",
    "consistency": "#2ecc71",
}
MODEL_COLORS = {
    "gemma3_4b": "#3498db",
    "llama3.1_8b": "#e74c3c",
}

WEIGHT_CONFIGS: List[Dict] = [
    {"name": "Baseline (Safety-priority)", "w_s": 0.40, "w_t": 0.35, "w_c": 0.25},
    {"name": "Safety-heavy", "w_s": 0.60, "w_t": 0.25, "w_c": 0.15},
    {"name": "Balanced", "w_s": 0.33, "w_t": 0.33, "w_c": 0.34},
    {"name": "Truthfulness-heavy", "w_s": 0.25, "w_t": 0.50, "w_c": 0.25},
    {"name": "Consistency-heavy", "w_s": 0.20, "w_t": 0.40, "w_c": 0.40},
]

DEFAULT_WEIGHTS = WEIGHT_CONFIGS[0]


def interpret_score(score: float) -> str:
    if score >= 0.80:
        return "\U0001f7e2 High"
    elif score >= 0.60:
        return "\U0001f7e1 Good"
    elif score >= 0.30:
        return "\U0001f7e0 Moderate"
    else:
        return "\U0001f534 Low"
