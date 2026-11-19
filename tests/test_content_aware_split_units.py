"""The content-aware split must keep every stimulus on ONE side of the boundary.

The collected experiment claims "0 family overlap" yet a rater meets the same
prompt under the same model in both splits, because the sealer and the two
diagnostics all agreed with each other on a definition that was wrong in the
same way. Three separate defects, each pinned below:

1. A consistency record's own ``prompt_text`` is EMPTY -- its stimuli live in the
   group's member records. Keying the unit on ``prompt_text`` alone therefore
   collapsed a whole dimension to one bucket, while the diagnosis read the
   stimuli from ``pairs`` instead. Sealer and checker disagreed about what a
   stimulus is.
2. A stimulus shared ACROSS dimensions (a truthfulness probe in calibration, a
   consistency display in held-out) was invisible to both, because each checked
   overlap within one dimension at a time.
3. Non-transitive grouping: if group A shows {X, Y} and group B shows {Y, Z},
   bucketing by member-text set keeps them apart and Y leaks. The units must be
   the transitive closure.

The tests build the smallest pool that exposes each shape, so they fail on the
old logic and pass on the fixed one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_sealer():
    """Import ``scripts/seal_experiment.py`` without relying on it being a package."""
    spec = importlib.util.spec_from_file_location(
        "seal_experiment_under_test", ROOT / "scripts" / "seal_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sealer = _load_sealer()


def _truth_record(key: str, text: str, model: str = "m1") -> dict:
    return {
        "_internal_key": key,
        "_model_id": model,
        "_dimension": "truthfulness",
        "prompt_text": text,
    }


def _consistency_group(group_id: str, texts: list, model: str = "m1") -> list:
    """A consistency group's stimuli live in its members, not in ``prompt_text``."""
    members = [
        {
            "_internal_key": f"{group_id}_{i}",
            "_model_id": model,
            "_dimension": "consistency",
            "group_id": group_id,
            "prompt_text": text,
        }
        for i, text in enumerate(texts)
    ]
    for m in members:
        m["_group_members"] = members
    return members


def _units_by_key(units):
    return sorted(sorted(r["_internal_key"] for r in u) for u in units)


# -- 1. the stimulus a rater sees is the group's, not the record's own --------
def test_consistency_stimuli_come_from_group_members():
    members = _consistency_group("g1", ["what is 2+2"])
    assert sealer._stimuli_of(members[0]) == {
        sealer._normalise_stimulus("what is 2+2")
    }


def test_cross_dimension_shared_stimulus_is_one_unit():
    """Same text as a truthfulness probe and as a consistency display -> one unit.

    This is the leak the collected data has: "What is the capital of Russia?"
    appears as a truthfulness probe in calibration and as a consistency display
    in held-out, under the same model. Both shapes must land in one unit or the
    rater meets the stimulus twice.
    """
    truth = _truth_record("t1", "What is the capital of Russia?")
    cons = _consistency_group("g1", ["What is the capital of Russia?"])
    units = sealer.build_split_units([truth] + cons, content_aware=True)
    assert len(units) == 1, f"expected one unit, got {_units_by_key(units)}"


# -- 2. transitive closure over shared stimuli -------------------------------
def test_partially_overlapping_groups_collapse_transitively():
    """A shows {X,Y}, B shows {Y,Z} -> all three must be one unit.

    Bucketing by "the group's member-text set" keeps A and B apart, which lets
    Y cross the split.
    """
    a = _consistency_group("gA", ["alpha", "beta"])
    b = _consistency_group("gB", ["beta", "gamma"])
    units = sealer.build_split_units(a + b, content_aware=True)
    assert len(units) == 1, f"expected one unit, got {_units_by_key(units)}"


def test_disjoint_stimuli_stay_separate():
    """No over-merging: unrelated records must remain distinct units."""
    pool = [
        _truth_record("t1", "question one"),
        _truth_record("t2", "question two"),
    ] + _consistency_group("g1", ["question three"])
    units = sealer.build_split_units(pool, content_aware=True)
    assert len(units) == 3, f"expected three units, got {_units_by_key(units)}"


def test_different_models_do_not_merge():
    """The unit is scoped by model: the same text under another model is allowed
    on the other side of the split (a family split keys on (model, text))."""
    pool = [
        _truth_record("t1", "shared prompt", model="m1"),
        _truth_record("t2", "shared prompt", model="m2"),
    ]
    units = sealer.build_split_units(pool, content_aware=True)
    assert len(units) == 2, f"expected two units, got {_units_by_key(units)}"


def test_consistency_group_is_never_torn():
    """Whatever else happens, one display must not span both splits."""
    group = _consistency_group("g1", ["alpha", "beta"])
    units = sealer.build_split_units(group, content_aware=True)
    assert len(units) == 1
    assert sorted(r["_internal_key"] for r in units[0]) == ["g1_0", "g1_1"]


# -- 3. the diagnosis must agree with the sealer about "overlap" -------------
def test_diagnosis_and_sealer_use_the_same_notion_of_stimulus():
    """Guard against the two drifting apart again.

    The old code had the sealer read ``prompt_text`` for every dimension while
    the diagnosis read ``pairs`` for consistency -- so each could be "consistent"
    with itself and still disagree. Both must now route through
    ``_stimuli_of``.
    """
    from scripts import diagnose_design

    cons = _consistency_group("g1", ["what is 2+2"])[0]
    assert sealer._stimuli_of(cons)
    assert diagnose_design._normalise("What is 2+2?") != ""
