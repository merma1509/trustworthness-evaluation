"""Confusion-matrix orientation must match its declared axes.

Bug this pins down: ``compute_agreement()`` builds ``{auto: {human: count}}``.
The inter-rater block called it as ``compute_agreement(a_labels, b_labels)``,
which makes RATER_B the *auto* axis — i.e. rows = RATER_B — but then declared
``confusion_axes = {"rows": "rater_a", "cols": "rater_b"}``. The label
contradicted the data, so every consumer read the matrix transposed.

A transposed confusion matrix is not cosmetic: for the safety dimension the
real disagreements are all (A=REFUSE, B=COMPLY), which appears in the
``COMPLY`` row. Read backwards it looks like A refused where B complied, which
inverts who was stricter.

These tests assert the declared axes against the actual data layout.
"""

from __future__ import annotations

from src.agreement import compute_agreement


def test_compute_agreement_rows_are_second_argument() -> None:
    """``compute_agreement(human, auto)`` puts ``auto`` on the rows."""
    human = ["REFUSE"] * 5
    auto = ["COMPLY"] * 5

    matrix = compute_agreement(human, auto)["confusion_matrix"]

    # auto is COMPLY -> that must be the row key.
    assert list(matrix) == ["COMPLY"], (
        f"expected rows=auto ('COMPLY'), got rows={list(matrix)}"
    )
    # human is REFUSE -> that must be the column key.
    assert matrix["COMPLY"] == {"REFUSE": 5}


def test_declared_axes_match_matrix_layout() -> None:
    """The declared axis names must describe the matrix that was produced.

    Reproduces the inter-rater construction: A passed as ``human``, B as
    ``auto``, so the matrix is {rater_b: {rater_a: count}}.
    """
    a_labels = ["REFUSE", "REFUSE", "COMPLY", "COMPLY"]
    b_labels = ["COMPLY", "COMPLY", "COMPLY", "COMPLY"]
    declared = {"rows": "rater_b", "cols": "rater_a"}

    matrix = compute_agreement(a_labels, b_labels)["confusion_matrix"]

    # Resolve the declared axis names back to the label sets they denote.
    axis_values = {"rater_a": a_labels, "rater_b": b_labels}

    row_labels = set(matrix)
    col_labels = {col for row in matrix.values() for col in row}

    assert row_labels <= set(axis_values[declared["rows"]]), (
        f"rows are {sorted(row_labels)} but declared rows="
        f"{declared['rows']!r} (labels {sorted(set(axis_values[declared['rows']]))})"
    )
    assert col_labels <= set(axis_values[declared["cols"]]), (
        f"cols are {sorted(col_labels)} but declared cols="
        f"{declared['cols']!r} (labels {sorted(set(axis_values[declared['cols']]))})"
    )


def test_transposed_axes_are_detectable_on_asymmetric_data() -> None:
    """A wrong declaration must be catchable, not merely plausible.

    Uses labels that appear on only one side so the transpose is unambiguous.
    """
    a_labels = ["REFUSE"] * 3      # rater A only ever refuses
    b_labels = ["COMPLY"] * 3      # rater B only ever complies

    matrix = compute_agreement(a_labels, b_labels)["confusion_matrix"]

    # Truthful orientation: B on rows.
    assert "COMPLY" in matrix, "rows must be RATER_B labels"
    assert matrix["COMPLY"] == {"REFUSE": 3}

    # The previously declared orientation would put A on the rows; assert the
    # data does NOT look like that, so the regression cannot silently return.
    a_on_rows = "REFUSE" in matrix and "COMPLY" not in matrix
    assert not a_on_rows, "matrix is built with RATER_B on the rows, not RATER_A"
