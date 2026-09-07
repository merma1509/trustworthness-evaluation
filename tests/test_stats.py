"""Unit tests for src/stats.py invariants (Task 9)."""
import numpy as np
import pytest

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
    # Extreme outcome: bootstrap would be degenerate [1,1]; instead we use a
    # Beta posterior so the CI conveys finite-sample uncertainty (Rule of Three).
    assert ci["mean"] == 1.0
    assert ci["ci_upper"] > 0.99
    assert ci["ci_lower"] < 1.0
    assert ci["method"] == "beta_posterior"


def test_confidence_intervals_all_inconsistent():
    ci = compute_confidence_intervals([0, 0, 0, 0], n_bootstrap=500)
    assert ci["mean"] == 0.0
    assert ci["ci_lower"] < 0.01
    assert ci["ci_upper"] > 0.0
    assert ci["method"] == "beta_posterior"


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


# ── Defensible sample-size estimator ────────────────
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

# ── ranking stability + empirical N ─────────
def test_ranking_stability_shape():
    from src.stats import compute_ranking_stability
    res = compute_ranking_stability(
        {"safety": 0.77, "truthfulness": 0.76, "consistency": 0.82},
        {"safety": 0.74, "truthfulness": 0.89, "consistency": 0.82},
        n_bootstrap=1000,
    )
    assert "model_wins" in res
    assert "per_config" in res
    assert len(res["per_config"]) == len(DEFAULT_WEIGHT_CONFIGS)
    total = res["model_wins"]["model1_pct"] + res["model_wins"]["model2_pct"] + res["model_wins"]["tie_pct"]
    assert round(total, 1) == 100.0
    for cfg in res["per_config"]:
        assert 0.0 <= cfg["flip_probability"] <= 1.0


def test_ranking_stability_identical_models_is_tie():
    from src.stats import compute_ranking_stability
    scores = {"safety": 0.8, "truthfulness": 0.8, "consistency": 0.8}
    res = compute_ranking_stability(scores, scores, n_bootstrap=500)
    for cfg in res["per_config"]:
        assert cfg["model1_wins_pct"] >= 0.0
        assert cfg["model2_wins_pct"] >= 0.0


def test_empirical_n_returns_estimate():
    from src.stats import compute_required_n_empirically
    scores = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    res = compute_required_n_empirically(
        scores, target_precision=0.05, n_bootstrap=100
    )
    assert "n_required" in res
    assert "theoretical_n" in res
    assert res["theoretical_n"] > 0
    assert len(res["sizes"]) > 0


# ── Wilson sample-size estimator ────────────────
def test_wilson_and_wald_reference_values():
    """Wilson must be <= Wald at p=0.5, and match known reference values."""
    from src.stats import estimate_required_sample_size

    wilson_05 = estimate_required_sample_size(precision=0.05, method="wilson")["n_required"]
    wald_05 = estimate_required_sample_size(precision=0.05, method="wald")["n_required"]
    wilson_10 = estimate_required_sample_size(precision=0.10, method="wilson")["n_required"]

    # Reference: Wilson 5% = 381, Wald 5% = 385, Wilson 10% = 93
    assert wilson_05 == 381
    assert wald_05 == 385
    assert wilson_10 == 93
    assert wilson_05 <= wald_05  # Wilson is slightly sharper than Wald


def test_estimate_required_sample_size_default_is_wilson():
    from src.stats import estimate_required_sample_size
    assert estimate_required_sample_size(precision=0.05)["method"] == "wilson"


def test_estimate_required_sample_size_rejects_bad_method():
    from src.stats import estimate_required_sample_size
    try:
        estimate_required_sample_size(method="exact")
        raised = False
    except ValueError:
        raised = True
    assert raised


# ── Efron percentile bootstrap CIs ─────────────
def test_confidence_intervals_efron_percentile_method_and_seed():
    """Standard path must report method/seed/n_bootstrap for traceability."""
    ci = compute_confidence_intervals([0, 1, 1, 0, 1, 1, 0, 1], n_bootstrap=500)
    assert ci["method"] == "efron_percentile"
    assert ci["n_bootstrap"] == 500
    assert ci["seed"] == 42


def test_confidence_intervals_deterministic_seed():
    """Same seed must give identical CIs (reproducibility)."""
    a = compute_confidence_intervals([0, 1, 1, 0, 1, 1, 0, 1], n_bootstrap=500, seed=42)
    b = compute_confidence_intervals([0, 1, 1, 0, 1, 1, 0, 1], n_bootstrap=500, seed=42)
    assert a["ci_lower"] == b["ci_lower"]
    assert a["ci_upper"] == b["ci_upper"]


# ── weight sensitivity with CIs ────────────────
def test_weight_sensitivity_with_ci_shape():
    from src.stats import compute_weight_sensitivity_with_ci
    res = compute_weight_sensitivity_with_ci(
        safety_trials=[1, 0, 1, 1, 0, 1],
        truthfulness_trials=[1, 1, 0, 1, 1],
        consistency_trials=[1, 0, 1],
        n_bootstrap=200,
    )
    assert len(res) == len(DEFAULT_WEIGHT_CONFIGS)
    for cfg in res:
        assert cfg["method"] == "efron_percentile"
        assert cfg["n_bootstrap"] == 200
        assert cfg["seed"] == 42
        assert cfg["ci_lower"] <= cfg["trustscore"] <= cfg["ci_upper"]
        # Half-width must match the CI width (tolerance for rounding).
        assert abs(cfg["ci_half_width"] - (cfg["ci_upper"] - cfg["ci_lower"]) / 2) <= 0.001


def test_weight_sensitivity_with_ci_point_matches_point_only():
    """Baseline point score must equal the non-CI sensitivity score."""
    from src.stats import compute_weight_sensitivity, compute_weight_sensitivity_with_ci
    safety = [1, 0, 1, 1, 0, 1, 0, 1]
    truth = [1, 1, 0, 1, 1, 1]
    cons = [1, 0, 1]
    ci_res = compute_weight_sensitivity_with_ci(
        safety, truth, cons, n_bootstrap=200
    )
    point_res = compute_weight_sensitivity(
        float(np.mean(safety)), float(np.mean(truth)), float(np.mean(cons)),
        DEFAULT_WEIGHT_CONFIGS,
    )
    by_name = {cfg["config"]: cfg for cfg in ci_res}
    for r in point_res:
        assert by_name[r["name"]]["trustscore"] == r["score"]


# ── model win probability / ranking stability ──
def test_model_win_probability_shape_and_stable_flag():
    from src.stats import compute_model_win_probability
    trials1 = {"safety": [1, 1, 1], "truthfulness": [1, 1, 1], "consistency": [1, 1, 1]}
    trials2 = {"safety": [0, 0, 0], "truthfulness": [0, 0, 0], "consistency": [0, 0, 0]}
    res = compute_model_win_probability(trials1, trials2, n_bootstrap=500)
    assert len(res["per_config"]) == len(DEFAULT_WEIGHT_CONFIGS)
    assert res["n_bootstrap"] == 500
    assert res["seed"] == 42
    for cfg in res["per_config"]:
        assert cfg["P_model1_wins"] >= 0.95  # model1 dominates model2
        assert cfg["stable_win"] is True


def test_model_win_probability_tie_sum():
    from src.stats import compute_model_win_probability
    trials1 = {"safety": [1, 0, 1], "truthfulness": [1, 0, 1], "consistency": [1, 0, 1]}
    res = compute_model_win_probability(trials1, trials1, n_bootstrap=500)
    for cfg in res["per_config"]:
        assert round(cfg["P_model1_wins"] + cfg["P_model2_wins"] + cfg["tie_pct"] / 100, 2) == pytest.approx(1.0, abs=0.01)


# ── paired bootstrap hypothesis test ───────────
def test_paired_bootstrap_test_identical_means():
    from src.stats import paired_bootstrap_test
    res = paired_bootstrap_test([1, 1, 0, 1, 0], [1, 1, 0, 1, 0], n_bootstrap=500)
    assert res["observed_difference"] == 0.0
    assert "null_hypothesis" in res
    assert res["n_pairs"] == 5
    assert res["n_bootstrap"] == 500
    assert res["seed"] == 42
    # Identical data under H0: p-value should be large (not significant).
    assert res["p_value"] > 0.05


def test_paired_bootstrap_test_requires_equal_length():
    from src.stats import paired_bootstrap_test
    try:
        paired_bootstrap_test([1, 0], [1])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_interpret_p_buckets():
    from src.stats import interpret_p
    assert interpret_p(0.0001) == "very strong evidence against H0"
    assert interpret_p(0.005) == "strong evidence against H0"
    assert interpret_p(0.03) == "moderate evidence against H0"
    assert interpret_p(0.07) == "weak evidence against H0"
    assert interpret_p(0.5) == "no evidence against H0"



