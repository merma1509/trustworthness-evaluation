"""Tests for src/agreement.py — Cohen's Kappa, weighted kappa, bootstrap CI.

These tests lock in the *corrected* agreement statistics so that any regression
to the earlier (incorrect) ``1/n_categories`` chance-agreement shortcut fails.
"""
import pytest

from src.agreement import (
    cohen_kappa,
    compute_agreement,
    kappa_bootstrap_ci,
)


def test_cohen_kappa_perfect_agreement():
    """Perfect agreement must yield kappa == 1."""
    confusion = {"a": {"a": 4}, "b": {"b": 2}}
    kappa, p_o, p_e = cohen_kappa(confusion)
    assert kappa == 1.0
    assert p_o == 1.0


def test_cohen_kappa_no_agreement():
    """When the raters never agree, kappa is zero (p_o == p_e == 0).
    Each rater always assigns a distinct single label, so observed agreement
    equals chance agreement."""
    confusion = {"a": {"b": 10}}
    kappa, p_o, p_e = cohen_kappa(confusion)
    assert p_o == 0.0
    assert p_e == 0.0
    assert kappa == 0.0


def test_cohen_kappa_single_category():
    kappa, p_o, p_e = cohen_kappa({"only": {"only": 5}})
    assert kappa == 1.0
    assert p_o == 1.0


def test_cohen_kappa_expected_from_marginals():
    """p_e must be computed from the marginals, not 1/n_categories."""
    confusion = {
        "consistent": {"consistent": 8, "inconsistent": 1},
        "correct": {"correct": 12},
        "inconsistent": {"consistent": 1},
        "incorrect": {"correct": 4, "incorrect": 4},
    }
    kappa, p_o, p_e = cohen_kappa(confusion)
    assert p_e == 0.34
    assert p_o == 0.8
    assert kappa == round((0.8 - 0.34) / (1 - 0.34), 4)


def test_cohen_kappa_unknown_labels_preserved():
    """Labels present in either rater (but not both) must be included in the
    diagonal-aligned matrix, giving kappa = 0.375 for this example."""
    confusion = {"a": {"a": 3}, "b": {"c": 2}}
    kappa, p_o, p_e = cohen_kappa(confusion)
    assert kappa == 0.375
    assert p_o == 0.6


def test_weighted_kappa_perfect_equals_one():
    kappa, _, _ = cohen_kappa({"low": {"low": 3}, "high": {"high": 2}}, weighted=True)
    assert kappa == 1.0


def test_weighted_kappa_symmetric_two_rater_agreement():
    """A symmetric disagreement in a 2x2 (or 3x3 aligned) matrix yields the
    same quadratic weighted kappa regardless of the distance of the single
    off-diagonal cell (here 0.6), matching scikit-learn."""
    near = cohen_kappa(
        {"low": {"low": 4, "mid": 1}, "mid": {"low": 1, "mid": 4}}, weighted=True
    )
    far = cohen_kappa(
        {"low": {"low": 4, "high": 1}, "high": {"low": 1, "high": 4}}, weighted=True
    )
    assert near[0] == 0.6
    assert near[0] == far[0]


def test_compute_agreement_requires_equal_length():
    with pytest.raises(ValueError):
        compute_agreement(["a", "b"], ["a"])


def test_compute_agreement_empty_input():
    res = compute_agreement([], [])
    assert res["n"] == 0
    assert res["cohens_kappa"] == 0.0


def test_compute_agreement_unlabelled_pairs_dropped():
    res = compute_agreement(["a", None, "b"], ["a", "b", "b"])
    assert res["n_valid_pairs"] == 2


def test_compute_agreement_includes_weighted_kappa():
    res = compute_agreement(["a", "b"], ["a", "b"])
    assert res["weighted_kappa"] == 1.0


def test_compute_agreement_with_ci():
    res = compute_agreement(
        ["a", "a", "b", "b", "a"],
        ["a", "a", "b", "a", "a"],
        with_ci=True,
        n_bootstrap=200,
    )
    assert "kappa_ci" in res
    ci = res["kappa_ci"]
    assert ci["ci_lower"] <= ci["kappa"] <= ci["ci_upper"]


def test_kappa_bootstrap_ci_empty():
    ci = kappa_bootstrap_ci([], [])
    assert ci["n"] == 0


def test_kappa_bootstrap_ci_perfect():
    ci = kappa_bootstrap_ci(["a", "a", "b", "b"], ["a", "a", "b", "b"])
    assert ci["kappa"] == 1.0
    assert ci["ci_lower"] == 1.0
    assert ci["ci_upper"] == 1.0


def test_kappa_bootstrap_ci_deterministic_seed():
    a = kappa_bootstrap_ci(
        ["a", "b", "c", "a", "b"], ["a", "b", "c", "a", "b"],
        random_seed=7, n_bootstrap=100,
    )
    b = kappa_bootstrap_ci(
        ["a", "b", "c", "a", "b"], ["a", "b", "c", "a", "b"],
        random_seed=7, n_bootstrap=100,
    )
    assert a == b
