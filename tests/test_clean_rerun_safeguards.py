"""Guarantees for the clean human-validation rerun (validation-v2).

These tests protect the two properties that make a rerun scientifically usable:

1. **Blindness** — a sealed template handed to a rater must not leak any
   ground-truth or identity field (auto label, model id, prompt id, attack
   type, expected behaviour, score, BEN_/PROMPT_ tokens). If a future raw
   schema change starts copying those fields into the template, the blindness
   gate must fail loudly at seal time.

2. **No silent dropping** — the agreement pipeline must account for *every*
   recorded annotation: each record is either scored (incl. adjudicated and
   triple-disagreement cases), explicitly unresolvable, or explicitly excluded
   with a reason. Hard cases must stay in the evaluation, never be discarded.

3. **Validation gate** — the one-command `validate-rerun` pipeline must fail
   closed when any integrity invariant breaks (coverage, κ/CI bounds, gold/auto
   join completeness).
"""

import copy
import json
from pathlib import Path

import pytest

from scripts.resolve_disagreements import resolve_disagreements
from scripts.seal_experiment import _assert_template_blind, _template_record
from scripts.validate_rerun_pipeline import (
    _check_agreement,
    _check_coverage,
    _check_gold_join,
)


# ── Blindness ──────────────────────────────────────────────────────────────
def _has_nonempty_field(node, name: str) -> bool:
    """True if ``node`` (dict/list/tree) has a NON-EMPTY value under ``name``."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == name and v not in ("", None) and v != [] and v != {}:
                return True
            if _has_nonempty_field(v, name):
                return True
    elif isinstance(node, list):
        return any(_has_nonempty_field(item, name) for item in node)
    return False


def _raw_rec(**overrides):
    """A raw-output record carrying all the sensitive fields raw files have."""
    rec = {
        "_dimension": "safety",
        "_internal_key": "K_0042",
        "prompt_text": "Do the thing now",
        "response": "I won't do that.",
        "prompt_id": "SAFE_0001",
        "model_id": "gemma3:4b",
        "attack_type": "jailbreak",
        "expected_behavior": "refuse",
        "auto_label": "COMPLY",
        "is_correct": False,
        "group_consistent": True,
    }
    rec.update(overrides)
    return rec


def test_template_strips_sensitive_fields():
    """Fields that reveal ground truth or identity must not reach the rater."""
    t = _template_record(_raw_rec())

    pr = t["prompt_record"]
    assert pr["prompt_id"] == ""
    assert pr["model_id"] == ""
    assert pr["attack_type"] == ""
    # The text the rater needs is present (the answer to judge)…
    assert pr["prompt_text"] == "Do the thing now"
    assert pr["model_response"] == "I won't do that."
    # …but nothing else leaked (non-empty sensitive field anywhere in the tree).
    for banned in ("auto_label", "is_correct", "expected_behavior",
                   "similarity", "score", "prompt_id", "attack_type", "model_id"):
        assert not _has_nonempty_field(t, banned), f"leaked field: {banned}"


def test_consistency_template_keeps_pairs_only():
    """Consistency templates expose the group pairs, never group_consistent."""
    crec = {
        "_dimension": "consistency",
        "_internal_key": "K_0007",
        "_group_members": [
            {"prompt_text": "P1", "response": "R1"},
            {"prompt_text": "P2", "response": "R2"},
        ],
        "group_consistent": True,
        "auto_label": "CONSISTENT",
        "model_id": "llama3.1:8b",
    }
    t = _template_record(crec)
    assert len(t["pairs"]) == 2
    assert t["prompts"] == ["P1", "P2"]
    assert t["responses"] == ["R1", "R2"]
    assert not _has_nonempty_field(t, "group_consistent")
    assert not _has_nonempty_field(t, "auto_label")


def test_assert_template_blind_rejects_injected_leaks():
    """The blindness gate must catch a future regression that copies a raw field."""
    t = _template_record(_raw_rec())

    # Simulate a future refactor: someone embeds the raw record verbatim.
    leaky = copy.deepcopy(t)
    leaky["prompt_record"]["prompt_id"] = "BEN_1234"
    with pytest.raises(RuntimeError):
        _assert_template_blind(leaky, "K_0042")

    leaky2 = copy.deepcopy(t)
    leaky2["extra"] = {"auto_label": "COMPLY"}
    with pytest.raises(RuntimeError):
        _assert_template_blind(leaky2, "K_0042")


def test_template_checksums_are_filled():
    """Every sealed template must carry a rubric + self-referential checksum.

    Guards ("Rubric checksums were empty"): a freshly sealed template
    must embed both the rubric revision the rater will follow and a per-record
    fingerprint so the rubric handed out — and the exact template a rater
    annotated — are provable from the record itself.
    """
    from scripts.seal_experiment import _record_checksum

    t = _template_record(_raw_rec(), rubric_checksum="abc123def456")

    md = t["metadata"]
    assert md["rubric_checksum"] == "abc123def456", "rubric checksum must propagate"
    tc = md["template_checksum"]
    assert len(tc) == 64, "template_checksum must be a full sha256 hex digest"
    # Self-referential: blanking the field then re-hashing reproduces the value,
    # so the fingerprint is acyclic and stable against the field itself.
    assert _record_checksum(t) == tc


def test_template_checksums_nonempty_when_rubric_file_hashed():
    """The default rubric-checksum path (from the real rubric file) is non-empty."""
    from pathlib import Path

    from src.sealing import sha256_file

    rubric_path = Path("docs/MANUAL_AUDIT_RUBRIC_v3.md")
    if not rubric_path.exists():
        pytest.skip("rubric file not present in this checkout")
    rubric_checksum = sha256_file(rubric_path)

    t = _template_record(_raw_rec(), rubric_checksum=rubric_checksum)
    assert t["metadata"]["rubric_checksum"] == rubric_checksum
    assert t["metadata"]["template_checksum"] != ""


# ── No silent dropping ────────────────────────────────────────────────────
def _ann(internal_key, dim, label, uncertain=False, response_invalid=False):
    """One rater annotation record in the sealed-template schema."""
    return {
        "internal_key": internal_key,
        "prompt_record": {
            "dimension": dim,
            "prompt_id": "",
            "prompt_text": "p",
            "model_response": "r",
        },
        "annotations": {
            f"{dim}_label": label,
            "label_source": "rater_judgment",
        },
        "flags": {
            "rater_uncertainty": "uncertain" if uncertain else "",
            "response_invalid": response_invalid,
        },
    }


def test_disagreement_cases_are_kept_in_evaluation(tmp_path):
    """Triple disagreement and adjudicated cases must survive, not be dropped."""
    recs_a = [_ann("K1", "truthfulness", "CORRECT_UNCERTAINTY")]
    recs_b = [_ann("K1", "truthfulness", "INCORRECT_FABRICATION")]
    recs_c = [_ann("K1", "truthfulness", "CORRECT_UNCERTAINTY")]  # matches A

    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    c_f = tmp_path / "c.jsonl"
    for f, recs in ((a_f, recs_a), (b_f, recs_b), (c_f, recs_c)):
        f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    report = resolve_disagreements([a_f], [b_f], [c_f])
    assert report["n_recorded"] == 1
    assert report["n_scored"] == 1
    assert report["n_unresolvable"] == 0
    assert report["n_excluded_response_invalid"] == 0
    out = report["outcomes"][0]
    assert out["outcome"] == "adjudicated_agreement_A"
    assert out["gold_label"] == "CORRECT_UNCERTAINTY"


def test_all_records_are_accounted_for_no_silent_drop(tmp_path):
    """Coverage: scored + unresolvable + excluded == every key.

    Mix of: agreement, disagreement resolved by C, an uncertain-only record
    (unresolvable), and an explicitly invalid response. None may silently
    disappear from the report.
    """
    # K1: A==B agreement
    # K2: A/B disagree, C agrees with B -> adjudicated
    # K3: all three raters uncertain -> no valid vote -> unresolvable
    # K4: response_invalid for all -> excluded (logged)
    recs_a = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "REFUSE"),
        _ann("K3", "safety", "COMPLY", uncertain=True),
        _ann("K4", "safety", "REFUSE", response_invalid=True),
    ]
    recs_b = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "COMPLY"),
        _ann("K3", "safety", "REFUSE", uncertain=True),
        _ann("K4", "safety", "COMPLY", response_invalid=True),
    ]
    recs_c = [
        _ann("K1", "safety", "REFUSE"),
        _ann("K2", "safety", "COMPLY"),
        _ann("K3", "safety", "COMPLY", uncertain=True),
        _ann("K4", "safety", "REFUSE", response_invalid=True),
    ]

    a_f = tmp_path / "a.jsonl"
    b_f = tmp_path / "b.jsonl"
    c_f = tmp_path / "c.jsonl"
    for f, recs in ((a_f, recs_a), (b_f, recs_b), (c_f, recs_c)):
        f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    report = resolve_disagreements([a_f], [b_f], [c_f])
    assert report["n_recorded"] == 4
    assert report["n_scored"] == 2            # K1, K2
    assert report["n_unresolvable"] == 1      # K3
    assert report["n_excluded_response_invalid"] == 1  # K4

    keys = {tuple(o["record_key"])[1] for o in report["outcomes"]}
    excluded_keys = {tuple(e["record_key"])[1] for e in report["excluded"]}
    assert keys | excluded_keys == {"K1", "K2", "K3", "K4"}
    # K4 must be present in the logged excluded list, never dropped silently.
    k4_excl = [e for e in report["excluded"] if tuple(e["record_key"])[1] == "K4"]
    assert len(k4_excl) == 1
    assert "response_invalid" in k4_excl[0]["reason"]


# ── Validation gate (validate_rerun_pipeline) ────────────────────────────
def _resolution_report(n_scored=3, n_unres=1, n_excl=1):
    """A resolution report shaped like resolve_disagreements output.

    coverage invariant: scored + unresolvable + excluded == recorded.
    """
    n_recorded = n_scored + n_unres + n_excl
    return {
        "n_recorded": n_recorded,
        "n_scored": n_scored,
        "n_unresolvable": n_unres,
        "n_excluded_response_invalid": n_excl,
        "outcome_counts": {},
        "gold_label_distribution": {},
    }


def test_validation_gate_coverage_ok():
    """A well-formed resolution report passes the no-silent-drop gate."""
    _check_coverage(_resolution_report(n_scored=3, n_unres=1, n_excl=1))


def test_validation_gate_coverage_fails_closed():
    """A report that drops a record (e.g. a future refactor) must abort."""
    good = _resolution_report(n_scored=3, n_unres=1, n_excl=1)
    good["n_scored"] = 2  # one record silently vanished
    with pytest.raises(SystemExit):
        _check_coverage(good)


def test_validation_gate_agreement_ok(tmp_path):
    """A sane agreement report (κ in bounds, n>0, CI contains κ) passes."""
    p = tmp_path / "agreement.json"
    p.write_text(json.dumps({
        "gold_vs_auto": {
            "overall": {
                "n": 20,
                "agreement_rate": 0.8,
                "cohens_kappa": 0.6,
                "kappa_ci": {"ci_lower": 0.4, "ci_upper": 0.8},
            },
            "per_dimension": {},
        },
        "inter_rater_A_vs_B": {
            "safety": {"cohens_kappa": 0.5, "agreement_rate": 0.75},
        },
    }))
    _check_agreement(p)


def test_validation_gate_agreement_fails_closed(tmp_path):
    """An out-of-bounds κ or zero n must abort the validation gate."""
    p = tmp_path / "agreement.json"
    p.write_text(json.dumps({
        "gold_vs_auto": {
            "overall": {"n": 0, "cohens_kappa": 2.0, "agreement_rate": 1.5},
            "per_dimension": {},
        },
        "inter_rater_A_vs_B": {},
    }))
    with pytest.raises(SystemExit):
        _check_agreement(p)


def test_validation_gate_gold_join_ok(tmp_path):
    """Gold records all carrying an auto label pass the join-completeness gate."""
    p = tmp_path / "gold.jsonl"
    recs = [
        {"internal_key": "K1", "auto_label": "REFUSE"},
        {"internal_key": "K2", "auto_label": "COMPLY"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    _check_gold_join(p)


def test_validation_gate_gold_join_fails_closed(tmp_path):
    """A gold record missing its auto label (join broken) must abort."""
    p = tmp_path / "gold.jsonl"
    p.write_text(
        json.dumps({"internal_key": "K1", "auto_label": None}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        _check_gold_join(p)


# ── Reproducible gold from plaintext auto labels (reproducibility) ────────
def _resolution_report_for_gold(tmp_path, n_scored=2):
    """A resolutions report image the real held-out shape.

    Emits `n_scored` A=B agreements, one per record_key ("dimension", "K_XXXX").
    """
    outcomes = []
    for i in range(1, n_scored + 1):
        ik = f"K_{i:04d}"
        outcomes.append({
            "record_key": ["safety", ik],
            "gold_label": "REFUSE",
            "outcome": "A=B_agreement",
            "label_A": "REFUSE",
            "label_B": "REFUSE",
            "label_C": None,
            "uncertain": {"RATER_A": False, "RATER_B": False, "ADJUDICATOR": False},
        })
    report = {
        "n_recorded": n_scored,
        "n_scored": n_scored,
        "n_unresolvable": 0,
        "n_excluded_response_invalid": 0,
        "outcomes": outcomes,
    }
    p = tmp_path / "resolutions.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def _run_generate_gold_labels(auto_file, resolutions_file, tmp_path):
    """Drive generate_gold_labels via its CLI and return (exit_code, gold path).

    Uses a subprocess (not an in-process main() call) so the exact Makefile
    invocation path is what the test guards.
    """
    import subprocess
    import sys

    out = tmp_path / "gold.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_gold_labels.py",
            "--resolutions", str(resolutions_file),
            "--auto-labels", str(auto_file),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    return proc.returncode, out


def test_gold_can_be_built_from_jsonl_plaintext_auto_labels(tmp_path):
    """The reproducible (no-passphrase) path must accept JSONL auto labels.

    A client (clone) without the sealed passphrase must be able to rebuild
    gold labels from the committed plaintext auto labels file. This guards the
    JSONL-reading branch in generate_gold_labels.
    """
    resolutions = _resolution_report_for_gold(tmp_path, n_scored=2)

    # Plaintext auto labels in the committed JSONL form (one record per line).
    auto_recs = [
        {"internal_key": "K_0001", "auto_label": "COMPLY",
         "model_id": "gemma3:4b", "attack_type": "benign",
         "expected_behavior": "comply"},
        {"internal_key": "K_0002", "auto_label": "REFUSE",
         "model_id": "llama3.1:8b", "attack_type": "jailbreak",
         "expected_behavior": "refuse"},
    ]
    auto_file = tmp_path / "auto_labels.jsonl"
    with auto_file.open("w", encoding="utf-8") as f:
        for r in auto_recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    rc, gold = _run_generate_gold_labels(auto_file, resolutions, tmp_path)
    assert rc == 0
    lines = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    by_key = {g["internal_key"]: g for g in lines}
    # auto labels were joined through (represents the gold-vs-auto comparison).
    assert by_key["K_0001"]["auto_label"] == "COMPLY"
    assert by_key["K_0002"]["auto_label"] == "REFUSE"
    assert by_key["K_0001"]["model_id"] == "gemma3:4b"
    # gold human label from the resolution report is preserved.
    assert by_key["K_0001"]["gold_human_label"] == "REFUSE"


def test_gold_from_jsonl_matches_committed_plaintext(tmp_path):
    """Rebuilding gold on a clone yields the same content as the committed gold.

    Uses the REAL committed plaintext auto_labels.jsonl and the REAL held-out
    resolution report, and asserts the rebuilt gold matches the committed
    experiment/gold/gold_labels.jsonl (proves end-to-end reproducibility from
    git without the passphrase).
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    exp_id = "trustworthiness-validation-2026-10-16-042935"
    resolutions = (
        repo_root / "experiment" / "agreements"
        / f"{exp_id}_heldout_disagreements.json"
    )
    auto_file = repo_root / "experiment" / "sealed" / "labels" / "auto_labels.jsonl"
    gold_committed = repo_root / "experiment" / "gold" / "gold_labels.jsonl"
    if not (auto_file.exists() and resolutions.exists() and gold_committed.exists()):
        import pytest
        pytest.skip("committed reproducibility artifacts not present in this checkout")

    out = tmp_path / "gold_rebuilt.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_gold_labels.py",
            "--resolutions", str(resolutions),
            "--auto-labels", str(auto_file),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0, proc.stderr

    def _sorted(path):
        return sorted(
            json.dumps(json.loads(l), sort_keys=True)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
        )

    assert _sorted(out) == _sorted(gold_committed)
