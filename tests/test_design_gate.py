"""The leakage detector must be part of the gate, and never silently green.

``diagnose_design`` detects the family-level leak (the same ``(model, prompt
text)`` appearing in calibration AND held-out) and exits non-zero. But nothing
in the pipeline invoked it, so ``validate_rerun_pipeline`` reported a leaking
split as clean — a detector that nobody calls is decoration.

Pinned here:

1. the validation run surfaces the design result (no silent skip);
2. a leaking split is a loud warning by default, because the already-collected
   experiment is known to leak and must stay reproducible;
3. ``--design-strict`` — the policy for any NEW experiment — fails closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import validate_rerun_pipeline as vrp  # noqa: E402


class _Args:
    """Minimal stand-in for the parsed CLI args (only ``design_strict`` is read)."""

    def __init__(self, design_strict: bool = False) -> None:
        self.design_strict = design_strict


def _patch_run(monkeypatch, rc: int) -> None:
    """Make the subprocess shim for diagnose_design return ``rc``."""
    def fake_run(cmd, cwd):
        assert any("diagnose_design.py" in str(part) for part in cmd), cmd
        return rc
    monkeypatch.setattr(vrp, "_run", fake_run)


def _run_gate(monkeypatch, rc: int, strict: bool) -> None:
    """Invoke just the design block through ``validate_rerun``'s helper."""
    _patch_run(monkeypatch, rc)
    vrp._check_design_leakage(_Args(design_strict=strict) if strict else _Args())


def test_clean_design_passes(monkeypatch):
    _run_gate(monkeypatch, rc=0, strict=True)  # must not raise


def test_leak_is_a_warning_by_default(monkeypatch, capsys):
    _run_gate(monkeypatch, rc=1, strict=False)  # must not raise
    out = capsys.readouterr().out
    assert "DESIGN WARNING" in out
    assert "design-strict" in out


def test_leak_fails_closed_under_strict(monkeypatch):
    with pytest.raises(SystemExit):
        _run_gate(monkeypatch, rc=1, strict=True)


def test_uninspected_split_is_not_green(tmp_path):
    """A split that was never inspected must never be reported as clean.

    A glob passed where a directory was expected, a typo'd --templates path, or
    an unsealed experiment all yield zero template files. The check used to
    return {"ok": True, "note": "no sealed templates present"}, so `diagnose-design`
    printed "no blockers" for data it had not opened — including for the r3
    re-seal, which in fact leaks. An unexamined split is not a verified split.
    """
    from scripts.diagnose_design import diagnose_textual_overlap

    res = diagnose_textual_overlap(tmp_path, tmp_path, None)

    assert res["ok"] is False, "missing templates must fail closed"
    assert res["n_calibration_files"] == 0
    assert res["n_heldout_files"] == 0
    assert "NOT inspected" in res["note"]
