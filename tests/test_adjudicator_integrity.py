"""An adjudication must be complete and justified, not merely present.

Bug this pins down: ``K_0070`` entered the adjudicator as
``INCORRECT_FABRICATION`` / ``uncertain``. A later commit blanked its label and
``rater_uncertainty``; a later one still re-emitted it as
``CORRECT_UNCERTAINTY`` / ``confident`` — the opposite verdict, presented as
confident, with no adjudication note. Nothing in the pipeline rejected it.

``cross_check_adjudicator`` only proved *which* records reached the
adjudicator, never that a verdict was actually reached. ``_check_adjudicator_
integrity`` closes that gap with two invariants that need no git history:

1. every record carries a non-empty dimension label and ``rater_uncertainty``;
2. a dispute where BOTH raters were uncertain must carry a note, because the
   adjudicator is then the only decision-maker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingest_annotations import _check_adjudicator_integrity

_KEY = ("truthfulness", "K_9999")
_SAFETY_KEY = ("safety", "K_8888")


def _adj(label: str = "CORRECT_UNCERTAINTY", uncertainty: str = "confident",
         notes: str = "") -> dict:
    return {
        _KEY: {
            "annotations": {"truthfulness_label": label},
            "flags": {"rater_uncertainty": uncertainty},
            "metadata": {"notes": notes},
        }
    }


def _rater(uncertainty: str) -> dict:
    return {
        _KEY: {
            "annotations": {},
            "flags": {"rater_uncertainty": uncertainty},
            "metadata": {"notes": ""},
        }
    }


def _call(adj, a=None, b=None) -> None:
    _check_adjudicator_integrity(
        adj,
        a if a is not None else _rater("confident"),
        b if b is not None else _rater("confident"),
        Path("ADJUDICATOR_test.jsonl"),
        "calibration",
    )


def test_complete_justified_adjudication_passes() -> None:
    _call(_adj(notes="Model is explicit that the planet is fictional"))


def test_blank_label_is_rejected() -> None:
    """A blank label is an unfilled template, not a decision."""
    with pytest.raises(ValueError, match="lack a complete, justified verdict"):
        _call(_adj(label=""))


def test_blank_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError, match="lack a complete, justified verdict"):
        _call(_adj(uncertainty=""))


def test_unexplained_resolution_of_double_uncertainty_is_rejected() -> None:
    """Both raters unsure + no note = a guess presented as a verdict."""
    with pytest.raises(ValueError, match="lack a complete, justified verdict"):
        _call(
            _adj(uncertainty="confident", notes=""),
            a=_rater("uncertain"),
            b=_rater("uncertain"),
        )


def test_explained_resolution_of_double_uncertainty_passes() -> None:
    _call(
        _adj(uncertainty="confident", notes="Fictional planet, nothing real misstated"),
        a=_rater("uncertain"),
        b=_rater("uncertain"),
    )


def test_note_not_required_when_a_rater_was_confident() -> None:
    """Only the double-uncertain case demands a note."""
    _call(
        _adj(uncertainty="confident", notes=""),
        a=_rater("confident"),
        b=_rater("uncertain"),
    )


def test_blanked_k_0070_shape_is_rejected() -> None:
    """The exact K_0070 intermediate state: label wiped, uncertainty wiped.

    Reproduces commit 9a62a1a, where the record survived with both fields empty.
    """
    broken = {
        _KEY: {
            "annotations": {"truthfulness_label": ""},
            "flags": {"rater_uncertainty": ""},
            "metadata": {"notes": ""},
        }
    }
    with pytest.raises(ValueError):
        _call(broken)


def test_safety_dimension_uses_its_own_label_field() -> None:
    """The completeness check must key on the right field per dimension."""
    adj = {
        _SAFETY_KEY: {
            "annotations": {"safety_label": ""},
            "flags": {"rater_uncertainty": "confident"},
            "metadata": {"notes": ""},
        }
    }
    with pytest.raises(ValueError):
        _check_adjudicator_integrity(
            adj, {}, {}, Path("ADJUDICATOR_test.jsonl"), "calibration"
        )
