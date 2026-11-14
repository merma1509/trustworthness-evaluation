"""A kappa that cannot express reliability must say so in the data, not only prose.

Bug this pins down: the kappa-paradox slices (truthfulness n=19, agreement
0.8947, kappa 0.0) shipped as plain ``"cohens_kappa": 0.0``. The number is
arithmetically correct — p_expected is 0.8947, so the denominator is 0.1053,
not zero — but a consumer that reads only the JSON sees "0.0" and concludes
chance-level agreement. The note explaining this lived in a separate string
field that nothing validated.

``kappa_interpretable`` (plus ``kappa_degenerate_reason``) makes the caveat
machine-readable without replacing a correct statistic with a null. Two
distinct degeneracies are flagged, because they are NOT the same defect:

* ``undefined_pe_zero``          -> kappa is None (0/0, single observed class);
* ``observed_equals_expected``   -> kappa is defined but carries no information.
"""

from __future__ import annotations

from src.agreement import compute_agreement


def test_skewed_marginals_flag_observed_equals_expected() -> None:
    """The kappa-paradox shape: kappa is 0.0, but it is *defined*.

    Reproduces the truthfulness slice: 17 agreements, 2 disagreements, one
    label dominating both marginals.
    """
    res = compute_agreement(
        ["CORRECT_UNCERTAINTY"] * 17 + ["INCORRECT_FABRICATION"] * 2,
        ["CORRECT_UNCERTAINTY"] * 19,
    )

    assert res["cohens_kappa"] == 0.0, "paradox kappa should stay 0.0, not become null"
    assert res["kappa_interpretable"] is False
    assert res["kappa_degenerate_reason"] == "observed_equals_expected"
    # The rate is the number to quote, and it must remain intact.
    assert res["agreement_rate"] == 0.8947


def test_single_observed_class_flags_undefined() -> None:
    """One label only: kappa is None because p_expected == 0 (0/0)."""
    res = compute_agreement(["CONSISTENT"] * 7, ["CONSISTENT"] * 7)

    assert res["cohens_kappa"] is None
    assert res["kappa_interpretable"] is False
    assert res["kappa_degenerate_reason"] == "undefined_pe_zero"


def test_genuine_agreement_is_interpretable() -> None:
    """A real chance-corrected result must stay flagged as interpretable.

    Safety slice shape: kappa 0.8197 on n=22 — healthy marginals.
    """
    res = compute_agreement(
        ["COMPLY"] * 10 + ["REFUSE"] * 10 + ["REFUSE"] * 2,
        ["COMPLY"] * 10 + ["REFUSE"] * 10 + ["COMPLY"] * 2,
    )

    assert res["kappa_interpretable"] is True
    assert res.get("kappa_degenerate_reason") is None
    assert res["cohens_kappa"] > 0.5


def test_balanced_chance_agreement_is_flagged() -> None:
    """Degeneracy is a property of the data, not of the sign of kappa.

    Here both raters split 50/50 and disagree half the time, so p_observed ==
    p_expected and kappa is 0.0 for the *same reason* as the paradox case —
    the coefficient cannot distinguish skill from chance.
    """
    res = compute_agreement(
        ["a"] * 10 + ["b"] * 10,
        ["a"] * 5 + ["b"] * 5 + ["a"] * 5 + ["b"] * 5,
    )

    assert res["cohens_kappa"] == 0.0
    assert res["kappa_interpretable"] is False


def test_flag_present_on_every_non_empty_slice() -> None:
    """Consumers must never have to guess: the key always exists."""
    for human, auto in (
        (["a", "b"], ["a", "b"]),
        (["a"] * 3, ["a"] * 3),
        (["a", "a", "b"], ["a", "b", "b"]),
    ):
        res = compute_agreement(human, auto)
        assert "kappa_interpretable" in res, f"missing flag for {human}/{auto}"
