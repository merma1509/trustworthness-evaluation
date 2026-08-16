"""Unit tests for src/stats.py invariants (Task 9)."""
import numpy as np

from src.stats import (
    DEFAULT_WEIGHT_CONFIGS,
    compute_confidence_intervals,
    compute_jackknife_stability,
    compute_paired_difference_ci,
    compute_weight_sensitivity,
)


def test_confidence_intervals_all_same():
    ci = compute_confidence_intervals([1, 1, 1, 1], n_bootstrap=500)
    assert ci["mean"] == 1.0
    assert ci["ci_lower"] == 1.0
    assert ci["ci_upper"] == 1.0


def test_confidence_intervals_contain_mean():
    ci = compute_confidence_intervals([0, 1, 1, 0, 1, 1, 0, 1], n_bootstrap=500)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]


def test_confidence_intervals_empty():
    ci = compute_confidence_intervals([])
    assert ci["n"] == 0
    assert ci["mean"] == 0.0


def test_paired_difference_ci_requires_equal_length():
    try:
        compute_paired_difference_ci([0, 1], [0])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_paired_difference_ci_symmetric():
    res = compute_paired_difference_ci(
        [1, 0, 1, 0], [0, 1, 0, 1], n_bootstrap=500
    )
    assert res["n_pairs"] == 4
    assert res["mean_difference"] == 0.0
    assert abs(res["p_value"] - 0.5) < 0.5  # not significantly better


def test_weight_sensitivity_matches_spec():
    weights = DEFAULT_WEIGHT_CONFIGS
    res = compute_weight_sensitivity(0.80, 0.70, 0.90, weights)
    cfgs = {r["name"]: r for r in res}
    # Baseline: 0.40*0.8 + 0.35*0.7 + 0.25*0.9
    assert cfgs["Baseline (Safety-priority)"]["score"] == round(
        0.40 * 0.80 + 0.35 * 0.70 + 0.25 * 0.90, 4
    )


def test_weight_configs_are_valid_probabilities():
    for w in DEFAULT_WEIGHT_CONFIGS:
        assert round(w["w_s"] + w["w_t"] + w["w_c"], 4) == 1.0


def test_jackknife_static_dataset():
    scores = [1] * 10
    jk = compute_jackknife_stability(scores, n_remove=1, n_iterations=100)
    assert jk["full_score"] == 1.0
    assert jk["std_jackknife"] == 0.0  # removing any prompt changes nothing


# ── Defensible sample-size estimator (Task 6) ────────────────


def test_sample_size_estimator_finite():
    from src.stats import estimate_required_sample_size
    res = estimate_required_sample_size(precision=0.05, confidence=0.95)
    assert res["n_required"] > 0
    assert np.isfinite(res["n_required"])


def test_sample_size_estimator_monotonic():
    """Tighter precision must require >= N (monotonic, not decreasing)."""
    from src.stats import estimate_required_sample_size
    loose = estimate_required_sample_size(precision=0.10)
    tight = estimate_required_sample_size(precision=0.05)
    assert tight["n_required"] >= loose["n_required"]


def test_sample_size_estimator_rejects_bad_args():
    from src.stats import estimate_required_sample_size
    try:
        estimate_required_sample_size(precision=1.5)
        raised = False
    except ValueError:
        raised = True
    assert raised
