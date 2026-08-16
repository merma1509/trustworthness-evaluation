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
    """Task 5: every audit record must carry a non-empty attack_type."""
    assert len(audit_records) > 0
    for rec in audit_records:
        assert rec.get("attack_type"), f"missing attack_type in {rec.get('audit_id')}"


def test_no_unknown_attack_type(audit_records):
    """Task 5: no record may fall back to 'unknown'."""
    for rec in audit_records:
        assert rec.get("attack_type") != "unknown", f"unknown in {rec.get('audit_id')}"


def test_consistency_groups_per_model():
    """Task 4: reference dataset has 11 multi-prompt consistency groups/model."""
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


# ── Blinded annotation schema (Task 7) ────────────────────────

BLINDED_DIR = PROJECT_ROOT / "results" / "audit" / "blinded"

LEAKED_KEYS = ("auto_label", "similarity", "human_label", "expected_behavior")


def test_blinded_annotation_has_no_leaked_fields():
    """The blinded annotation must not expose auto_label / similarity / labels."""
    if not BLINDED_DIR.exists():
        pytest.skip("blinded annotation dir not present")
    for path in BLINDED_DIR.glob("blinded_annotation_*.jsonl"):
        for line in path.open():
            rec = json.loads(line)
            for key in LEAKED_KEYS:
                assert key not in rec, f"leaked '{key}' in {path.name}"

            if "response" in rec:
                assert "auto_label" not in str(rec["response"])
