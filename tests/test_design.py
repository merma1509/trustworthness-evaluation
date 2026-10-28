"""C2/C3 tests: model-scoped displays + content-aware family split.

These cover the design fixes prompted by the independent review:

C2 — model-scoping in consistency displays:
      every display is backed by exactly one model, tracked under
      ``metadata.display_model`` (never shown to the rater).

C3 — family-level (content-aware) calibration/held-out split:
      the indivisible unit is (model, normalised prompt_text), so no stimulus
      text can leak across the split even under different internal keys.
"""

from __future__ import annotations

import json


def _template_record_from_members(model_id, members, group_id="group_1"):
    """Build a raw record the way _build_response_pool + member-linking do."""
    from scripts.seal_experiment import _template_record

    rec = {
        "_dimension": "consistency",
        "_internal_key": f"K_{model_id.split(':')[0][:3]}_1",
        "_model_id": model_id,
        "group_id": group_id,
        "_group_members": members,
        "group_consistent": True,
        "auto_label": "CONSISTENT",
    }
    return _template_record(rec)


def _has_nonempty_field(node, name: str) -> bool:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == name and v not in ("", None) and v != [] and v != {}:
                return True
            if _has_nonempty_field(v, name):
                return True
    elif isinstance(node, list):
        return any(_has_nonempty_field(item, name) for item in node)
    return False


# ── C2: model-scoping of consistency displays ───────────────────────────────
def test_display_model_is_traced_in_metadata():
    """Metadata must record the single backing model of a consistency display."""
    t = _template_record_from_members(
        "gemma3:4b",
        [
            {"prompt_text": "What is 2+2?", "response": "4"},
            {"prompt_text": "What is 2+2?", "response": "4"},
        ],
    )
    assert t["metadata"]["display_model"] == "gemma3:4b"
    # The rater must stay blind to the model: no top-level model_id leak.
    assert not _has_nonempty_field(t["prompt_record"], "model_id")
    assert not _has_nonempty_field(t["pairs"], "model_id")
    assert "display_model" in t["metadata"]


def test_display_model_is_per_model_even_for_shared_group_id():
    """Two models sharing a descriptive group_id keep separate displays."""
    g = _template_record_from_members(
        "gemma3:4b",
        [{"prompt_text": "G", "response": "g-r1"}],
        group_id="group_1",
    )
    l = _template_record_from_members(
        "llama3.1:8b",
        [{"prompt_text": "L", "response": "l-r1"}],
        group_id="group_1",
    )
    assert g["metadata"]["display_model"] == "gemma3:4b"
    assert l["metadata"]["display_model"] == "llama3.1:8b"
    assert g["responses"] == ["g-r1"]
    assert l["responses"] == ["l-r1"]
    # Visible pairs stay anonymous; only metadata carries the model.
    assert not _has_nonempty_field(g, "model_id")


# ── C3: content-aware family split ──────────────────────────────────────────
def test_family_unit_key_groups_shared_text():
    """Records with identical normalised prompt text share a split unit."""
    from scripts.seal_experiment import (
        _family_unit_key,
        _normalise_stimulus,
    )

    a = {"_dimension": "truthfulness", "prompt_text": "What is 2 + 2?"}
    b = {"_dimension": "truthfulness", "prompt_text": "What is 2+2 ?"}
    c = {"_dimension": "truthfulness", "prompt_text": "What is 2 - 2 ?"}

    assert _normalise_stimulus(a["prompt_text"]) == _normalise_stimulus(
        b["prompt_text"]
    )
    assert _family_unit_key(a, "gemma3:4b") == _family_unit_key(b, "gemma3:4b")
    # Genuinely different text -> different unit even for the same model.
    assert _family_unit_key(a, "gemma3:4b") != _family_unit_key(c, "gemma3:4b")
    # Different model with same text -> different unit (family = model+text).
    assert _family_unit_key(a, "gemma3:4b") != _family_unit_key(a, "llama3.1:8b")


def test_content_aware_split_has_no_text_overlap(tmp_path):
    """A content-aware split must leave zero prompt-text overlap.

    Mimic a tiny pool where the same truthfulness prompt text appears twice
    under two different prompt ids (K_0001 / K_0002). With content-aware units
    both records must land in ONE split (never torn), so textual overlap is 0.
    """
    import scripts.seal_experiment as seal

    raw = tmp_path / "raw"
    raw.mkdir()
    shared_text = "What is the capital of Russia?"
    rows = [
        {"prompt_id": "T1", "prompt_text": shared_text,
         "response": "Moscow", "group_id": "", "dimension": "truthfulness"},
        {"prompt_id": "T2", "prompt_text": shared_text,
         "response": "Moscow", "group_id": "", "dimension": "truthfulness"},
    ]
    (raw / "gemma3_4b_truthfulness.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )

    pool = seal._build_response_pool(raw)
    # run the split with content_aware=True via the same code path used in main()
    units = seal.build_split_units(pool, True)
    # All records sharing (model, text) must be in the SAME unit.
    assert len(units) == 1, "shared text must collapse into one family unit"
