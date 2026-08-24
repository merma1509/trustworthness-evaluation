"""Integration tests for the audit pipeline (Task 4 group counts + Task 5
attack_type propagation).
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"
AUDIT_PATH = PROJECT_ROOT / "results" / "audit" / "all_audit.jsonl"


@pytest.fixture(scope="module")
def audit_records():
    if not AUDIT_PATH.exists():
        pytest.skip("all_audit.jsonl not present")
    with AUDIT_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_no_audit_record_missing_attack_type(audit_records):
    """Task 5: every audit record must carry a non-empty attack_type"""
    assert len(audit_records) > 0
    for rec in audit_records:
        assert rec.get("attack_type"), f"missing attack_type in {rec.get('audit_id')}"


def test_no_unknown_attack_type(audit_records):
    """Task 5: no record may fall back to 'unknown' attack type"""
    for rec in audit_records:
        assert rec.get("attack_type") != "unknown", f"unknown in {rec.get('audit_id')}"


def test_consistency_groups_per_model():
    """Task 4: reference dataset has 11 multi-prompt consistency groups/model"""
    model = "gemma3_4b"
    path = RAW_OUTPUTS_DIR / f"{model}_consistency.jsonl"
    if not path.exists():
        pytest.skip("consistency raw outputs not present")
    groups = set()
    singleton_groups = set()
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            gid = rec.get("group_id")
            if rec.get("is_singleton"):
                singleton_groups.add(gid)
            else:
                groups.add(gid)
    assert len(groups) == 11, f"expected 11 multi-prompt groups, got {len(groups)}"
    assert len(singleton_groups) == 5


# ── Blinded annotation schema ────────────────────────
BLINDED_DIR = PROJECT_ROOT / "results" / "audit" / "blinded"

LEAKED_KEYS = ("auto_label", "similarity", "human_label", "expected_behavior")


def test_blinded_annotation_has_no_leaked_fields():
    """The blinded annotation must not expose auto_label / similarity / labels / expected behavior"""
    if not BLINDED_DIR.exists():
        pytest.skip("blinded annotation dir not present")
    for path in BLINDED_DIR.glob("blinded_annotation_*.jsonl"):
        for line in path.open():
            rec = json.loads(line)
            for key in LEAKED_KEYS:
                assert key not in rec, f"leaked '{key}' in {path.name}"

            if "response" in rec:
                assert "auto_label" not in str(rec["response"])


def _load_blinded_consistency_groups():
    """Yield every consistency group record across the blinded files"""
    if not BLINDED_DIR.exists():
        return
    for path in BLINDED_DIR.glob("blinded_annotation_*.jsonl"):
        for line in path.open():
            rec = json.loads(line)
            if rec.get("dimension") == "consistency":
                yield rec


def test_blinded_consistency_no_duplicate_group_members():
    """Regression: a consistency group must never list the same prompt twice

    The original ``_load_raw_lookup`` keyed every consistency row under BOTH
    ``prompt_id`` and ``group_id``, so ``source.values()`` contained the same
    record twice and groups in the blinded output ended up with duplicated
    members (e.g. ``CON_027`` appearing 3x in ``group_11``)

    For a consistency group the member prompt ids (and prompt texts) must be
    unique -- no member may be dropped nor duplicated
    """
    if not BLINDED_DIR.exists():
        pytest.skip("blinded annotation dir not present")

    seen = False
    for rec in _load_blinded_consistency_groups():
        seen = True
        pairs = rec.get("pairs", [])
        markers = [(p.get("prompt_id"), p.get("prompt")) for p in pairs]
        assert len(markers) == len(set(markers)), (
            f"duplicate group member in {rec.get('audit_id')}: {markers}"
        )
    assert seen, "no consistency records found in blinded annotation"


def test_blinded_consistency_members_match_raw_outputs():
    """Each blinded consistency group's members must equal the raw-output members

    This guards against the OTHER failure mode of the same bug: after the first
    (incomplete) fix that keyed groups by a single record, members were silently
    DROPPED (e.g. ``group_11`` showing only ``CON_027`` and losing ``CON_026``)
    The set of prompt ids in every blinded group must exactly match the
    reference group membership from the raw consistency file
    """
    if not BLINDED_DIR.exists():
        pytest.skip("blinded annotation dir not present")

    # Build raw membership reference: (model, group_id) -> set(prompt_ids)
    raw_ref = {}
    for path in RAW_OUTPUTS_DIR.glob("*_consistency.jsonl"):
        model_key = path.stem.rsplit("_", 1)[0]  # e.g. gemma3_4b
        for line in path.open():
            rec = json.loads(line)
            gid = rec.get("group_id") or rec.get("prompt_id")
            raw_ref.setdefault((model_key, gid), set()).add(rec.get("prompt_id"))

    for rec in _load_blinded_consistency_groups():
        audio_id = rec.get("audit_id", "")
        # model key is the audit-id prefix before "_cons_"
        model_key = audio_id.split("_cons_")[0]
        gid = rec.get("group_id")
        blinded_members = {p.get("prompt_id") for p in rec.get("pairs", [])}
        raw_members = raw_ref.get((model_key, gid), set())

        # Only groups present in raw are validated (singletons are excluded
        # from the audit and will have an empty raw multi-prompt membership).
        if raw_members:
            assert blinded_members == raw_members, (
                f"{audio_id}: blinded members {blinded_members} != "
                f"raw members {raw_members}"
            )
