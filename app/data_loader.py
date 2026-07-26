"""Load all evaluation results, scores, and raw outputs for the dashboard."""
import json
from typing import Dict, List, Optional

from app.config import MODEL_NAMES, RAW_OUTPUTS_DIR, RESULTS_DIR


def load_scores(model_key: str) -> Optional[Dict]:
    path = RESULTS_DIR / model_key / "scores.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.loads(f.read().strip().rstrip(","))


def load_confidence_intervals(model_key: str) -> Optional[Dict]:
    path = RESULTS_DIR / model_key / "confidence_intervals.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.loads(f.read().strip().rstrip(","))


def load_weight_sensitivity(model_key: str) -> Optional[List[Dict]]:
    path = RESULTS_DIR / model_key / "weight_sensitivity.json"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_outputs(model_key: str, dimension: str) -> List[Dict]:
    path = RAW_OUTPUTS_DIR / f"{model_key}_{dimension}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_scores() -> Dict[str, Dict]:
    scores = {}
    for model_key in MODEL_NAMES:
        data = load_scores(model_key)
        if data is not None:
            scores[model_key] = data
    return scores


def load_all_confidence_intervals() -> Dict[str, Dict]:
    cis = {}
    for model_key in MODEL_NAMES:
        data = load_confidence_intervals(model_key)
        if data is not None:
            cis[model_key] = data
    return cis


def load_all_weight_sensitivity() -> Dict[str, List[Dict]]:
    sensitivities = {}
    for model_key in MODEL_NAMES:
        data = load_weight_sensitivity(model_key)
        if data is not None:
            sensitivities[model_key] = data
    return sensitivities


def load_ranking_stability() -> Optional[Dict]:
    path = RESULTS_DIR / "ranking_stability.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.loads(f.read().strip().rstrip(","))


def load_manual_audit() -> List[Dict]:
    path = RESULTS_DIR / "manual_audit_consistency.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_manual_audit(records: List[Dict]):
    path = RESULTS_DIR / "manual_audit_consistency.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def get_dimension_score(scores: Dict, dimension: str) -> float:
    return scores.get("dimension_scores", {}).get(dimension, {}).get("score", 0.0)


def get_confusion_matrix(scores: Dict) -> Dict:
    return (
        scores.get("dimension_scores", {})
        .get("safety", {})
        .get("confusion_matrix", {})
    )


def models_available() -> List[str]:
    available = []
    for model_key in MODEL_NAMES:
        if (RESULTS_DIR / model_key / "scores.json").exists():
            available.append(model_key)
    return available


def compute_trustscore(
    s: float, t: float, c: float,
    w_s: float = 0.40, w_t: float = 0.35, w_c: float = 0.25
) -> float:
    return round(w_s * s + w_t * t + w_c * c, 4)
