"""Guards that the declaration badge reflects the file instead of guessing.

The bug these tests pin down: ``app/tabs/rater_annotation._declaration_signed``
tested for the *absence* of escaped-underscore runs (``\\_\\_\\_\\_``) on the
assumption that ``onboard_raters.declaration_form()`` emitted escaped blanks.
It actually emits plain underscores (``Date: __________``), so the marker was
never present, ``"\\_\\_\\_\\_" not in md_text`` was always ``True``, and a
completely blank form was reported as ``[SIGNED]``.

That mattered because all three declaration forms in
``experiment/rater_declarations`` were blank while the UI claimed otherwise —
an independence/consent artifact that looked satisfied but was not.

The properties protected here:

1. A blank generated form reads as NOT signed.
2. A form with all three binding fields reads as signed.
3. Any partially filled form stays unsigned (never optimistically signed).
4. Every rater's on-disk declaration is reported honestly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tabs.rater_annotation import (
    _declaration_path,
    _declaration_signed,
    _declaration_values,
    _fill_declaration,
)

EXPERIMENT_ID = "trustworthiness-validation-2026-10-16-042935"
RATERS = ("RATER_A", "RATER_B", "ADJUDICATOR")

# A minimal stand-in for onboard_raters.declaration_form(), using the same
# PLAIN-underscore blanks the real generator emits. This is the exact input the
# old heuristic misjudged.
BLANK_FORM = """# RATER DECLARATION FORM — RATER_A

**Experiment:** trustworthiness-validation-2026-10-16-042935

I, ____________________________, agree to the following:

1. **Independence**: I have NOT seen the model responses I will annotate.

---

Rater ID: RATER_A

Date: __________

Signature: __________
"""


def test_blank_generated_form_is_not_signed():
    """The plain-underscore blank the real generator emits must read unsigned."""
    assert "____" in BLANK_FORM                        # plain blanks
    assert "\\_\\_\\_\\_" not in BLANK_FORM              # the marker old code wanted
    assert _declaration_signed(BLANK_FORM) is False


def test_fully_filled_form_is_signed():
    filled = _fill_declaration(
        BLANK_FORM, name="Alex Kim", date="2026-11-13", signature="A. Kim"
    )
    assert _declaration_signed(filled) is True
    assert _declaration_values(filled) == {
        "name": "Alex Kim",
        "date": "2026-11-13",
        "signature": "A. Kim",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Alex Kim", "date": "", "signature": ""},            # name only
        {"name": "Alex Kim", "date": "2026-11-13", "signature": ""},  # no signature
        {"name": "", "date": "2026-11-13", "signature": "A. Kim"},    # no name
        {"name": "", "date": "", "signature": "A. Kim"},              # sig only
    ],
)
def test_partially_filled_form_stays_unsigned(kwargs):
    """Never report an incomplete declaration as signed."""
    partial = _fill_declaration(BLANK_FORM, **kwargs)
    assert _declaration_signed(partial) is False


def test_empty_string_is_not_signed():
    assert _declaration_signed("") is False


@pytest.mark.parametrize("rater", RATERS)
def test_badge_matches_the_on_disk_declaration(rater):
    """The badge must agree with the fields actually stored for each rater.

    This is the regression that would have caught the original bug: the forms
    are blank, so the badge must say so.
    """
    path = Path(_declaration_path(EXPERIMENT_ID, rater))
    if not path.exists():
        pytest.skip(f"no declaration form on disk for {rater}")

    md_text = path.read_text(encoding="utf-8")
    values = _declaration_values(md_text)
    expected = all(bool(values[f]) for f in ("name", "date", "signature"))
    assert _declaration_signed(md_text) is expected
