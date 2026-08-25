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
GROUND_TRUTH_PATH = BLINDED_DIR / "ground_truth_blinded.json"

LEAKED_KEYS = ("auto_label", "similarity", "human_label", "expected_behavior")


def _load_ground_truth():
    """Return {(anon_id): [consistency_entries]} from the analyst-only ground truth.

    The blinded JSONL deliberately strips audit_id / group_id / prompt_id so the
    annotator cannot reverse-engineer the real prompt or its auto label. To map a
    blinded consistency record back to its real (model, group) membership we use
    the analyst-only ``ground_truth_blinded.json`` (never handed to annotators).
    """
    if not GROUND_TRUTH_PATH.exists():
        return {}
    doc = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    by_anon = {}
    for entries in doc.get("by_anon_id", {}).values():
        for e in entries:
            if e.get("anon_id") and e.get("dimension") == "consistency":
                by_anon.setdefault(e["anon_id"], []).append(e)
    return by_anon


def _resolve_consistency_group(rec):
    """Resolve a blinded consistency record to (model, group_id, attack_type).

    The blinded ``rec`` only carries ``anon_id`` and ``dimension``. We look up its
    real audit id in the analyst-only ground truth and parse the consistency group
    from the audit_id (e.g. ``gemma3_4b_cons_group_11`` -> ``gemma3_4b``, ``group_11``).
    """
    for entry in _load_ground_truth().get(rec.get("anon_id"), []):
        aid = entry.get("audit_id", "")
        for model in ("gemma3_4b", "llama3.1_8b"):
            prefix = model + "_cons_"
            if aid.startswith(prefix):
                return model, aid[len(prefix):], entry.get("attack_type")
    return None


def _raw_consistency_members(model):
    """Return {(group_id): list of (prompt_id, prompt_text)} from raw outputs."""
    path = RAW_OUTPUTS_DIR / f"{model}_consistency.jsonl"
    members = {}
    if not path.exists():
        return members
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            gid = rec.get("group_id")
            if gid:
                members.setdefault(gid, []).append(
                    (rec.get("prompt_id"), rec.get("prompt_text"))
                )
    return members


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
    """Regression: a consistency group must never list the same prompt twice.

    The original ``_load_raw_lookup`` keyed every consistency row under BOTH
    ``prompt_id`` and ``group_id``, so ``source.values()`` contained the same
    record twice and groups in the blinded output ended up with duplicated
    members (e.g. ``CON_027`` appearing 3x in ``group_11``).

    The blinded generator strips ``prompt_id``, so duplicates are detected via the
    prompt *text*. Subtlety: **perturbation** groups hold distinct re-phrasings
    (repeated text = the duplication bug), whereas **repetition** groups
    intentionally repeat the same prompt text — for those we assert the member
    *count* matches the raw group (nothing dropped/inflated) instead of naive
    text uniqueness.
    """
    if not BLINDED_DIR.exists() or not _load_ground_truth():
        pytest.skip("blinded annotation dir/ground truth not present")

    seen = False
    for rec in _load_blinded_consistency_groups():
        seen = True
        resolved = _resolve_consistency_group(rec)
        assert resolved is not None, (
            f"cannot resolve consistency group for anon_id {rec.get('anon_id')}"
        )
        model, gid, attack_type = resolved

        texts = [p.get("prompt", "") for p in rec.get("pairs", [])]
        raw_members = _raw_consistency_members(model).get(gid, [])

        if attack_type == "perturbation":
            # Distinct re-phrasings: any repeated text is the duplication bug.
            assert len(texts) == len(set(texts)), (
                f"duplicate group member in {model} {gid}: {texts}"
            )
        # Every group (incl. repetition) must preserve the raw member count:
        # guards against members being dropped OR inflated.
        assert len(texts) == len(raw_members), (
            f"{model} {gid}: blinded has {len(texts)} members but raw has "
            f"{len(raw_members)} ({rec.get('anon_id')})"
        )
    assert seen, "no consistency records found in blinded annotation"


def test_blinded_consistency_members_match_raw_outputs():
    """Each blinded consistency group's members must equal the raw-output members.

    This guards against the OTHER failure mode of the same bug: after the first
    (incomplete) fix that keyed groups by a single record, members were silently
    DROPPED (e.g. ``group_11`` showing only ``CON_027`` and losing ``CON_026``).

    The blinded generator strips ``prompt_id``, so membership equality is checked
    on the set of prompt *texts* against the raw consistency file (resolution via
    the analyst-only ground truth).
    """
    if not BLINDED_DIR.exists() or not _load_ground_truth():
        pytest.skip("blinded annotation dir/ground truth not present")

    for rec in _load_blinded_consistency_groups():
        resolved = _resolve_consistency_group(rec)
        assert resolved is not None, (
            f"cannot resolve consistency group for anon_id {rec.get('anon_id')}"
        )
        model, gid, _attack_type = resolved

        blinded_texts = {p.get("prompt", "") for p in rec.get("pairs", [])}
        raw_texts = {
            text for _pid, text in _raw_consistency_members(model).get(gid, [])
        }

        # Only groups present in raw are validated (singletons are excluded
        # from the audit and have no multi-prompt membership).
        assert raw_texts, f"{model} {gid}: no raw membership found"
        assert blinded_texts == raw_texts, (
            f"anon_id {rec.get('anon_id')} ({model} {gid}): blinded members "
            f"{blinded_texts} != raw members {raw_texts}"
        )
