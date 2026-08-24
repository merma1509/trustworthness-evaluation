"""Tests for the Error Heatmap helper."""
import pytest

from scripts.error_heatmap import (
    _build_confusion_matrix,
    _build_dimension_matrix,
)

SAMPLE_AGREEMENT = {
    "overall": {
        "confusion_matrix": {
            "correct": {"correct": 17, "incorrect": 2},
            "incorrect": {"correct": 1, "incorrect": 1},
            "consistent": {"consistent": 8},
            "inconsistent": {"inconsistent": 1},
        }
    },
    "by_dimension": {
        "safety": {"n": 10, "agreement_rate": 0.9},
        "truthfulness": {"n": 10, "agreement_rate": 0.9},
        "consistency": {"n": 10, "agreement_rate": 0.9},
    },
}

SAMPLE_VALIDATION = {
    "rq1_agreement": {
        "by_dimension": {
            "safety": {"cohens_kappa": 0.615},
            "truthfulness": {"cohens_kappa": 0.0},
            "consistency": {"cohens_kappa": 0.615},
        }
    }
}


def test_dimension_matrix_includes_dimensions():
    rows = _build_dimension_matrix(SAMPLE_AGREEMENT, SAMPLE_VALIDATION)
    dims = {r["dimension"] for r in rows}
    assert dims == {"Safety", "Truthfulness", "Consistency"}


def test_dimension_matrix_reads_kappa_from_validation():
    rows = _build_dimension_matrix(SAMPLE_AGREEMENT, SAMPLE_VALIDATION)
    by_dim = {r["dimension"]: r for r in rows}
    assert by_dim["Truthfulness"]["kappa"] == pytest.approx(0.0)
    assert by_dim["Safety"]["kappa"] == pytest.approx(0.615)


def test_dimension_matrix_falls_back_to_agreement_kappa():
    # No validation report -> kappa should come from the agreement report, else 0.
    rows = _build_dimension_matrix(SAMPLE_AGREEMENT, {})
    by_dim = {r["dimension"]: r for r in rows}
    assert by_dim["Safety"]["kappa"] == 0.0  # agreement report has no kappa here


def test_confusion_matrix_square():
    labels, matrix = _build_confusion_matrix(SAMPLE_AGREEMENT)
    n = len(labels)
    assert matrix is not None
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)


def test_confusion_matrix_totals():
    labels, matrix = _build_confusion_matrix(SAMPLE_AGREEMENT)
    total = sum(sum(row) for row in matrix)
    # 17+2 + 1+1 + 8 + 1 = 30
    assert total == 30


def test_missing_confusion_returns_none():
    labels, matrix = _build_confusion_matrix({})
    assert labels is None and matrix is None
