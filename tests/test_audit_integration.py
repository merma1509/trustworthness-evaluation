"""Integration tests for the audit pipeline (group counts +
attack_type propagation) and the sealed multi-rater dashboard contract.
"""
import json
import sys
from pathlib import Path
from unittest import mock

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


class _FakeContainer:
    """Minimal Streamlit column that participates in ``with`` blocks."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSt:
    """In-memory stub capturing metrics so we can assert on the dashboard values."""

    def __init__(self):
        self.metrics = []

    # UI calls that render without data we assert on.
    def markdown(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass

    def code(self, *a, **k):
        pass

    def dataframe(self, *a, **k):
        pass

    def columns(self, n):
        return [_FakeContainer() for _ in range(n)]

    def metric(self, label, value, **kw):
        self.metrics.append((label, value))


def test_dashboard_render_blocks_read_sealed_schema():
    """The dashboard's rater + gold-vs-auto blocks consume the sealed report
    schema (inter_rater_A_vs_B per dimension, gold_vs_auto overall/per_dimension)
    and emit the expected κ metrics without exception."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import app.tabs.inter_annotator as ia

    st = _FakeSt()
    ia._render_rater_block(
        st,
        {
            "safety": {"n": 16, "cohens_kappa": 1.0, "agreement_rate": 0.875},
            "truthfulness": {"n": 24, "cohens_kappa": 0.0, "agreement_rate": 0.958},
            "consistency": {"n": 23, "cohens_kappa": 0.0, "agreement_rate": 0.739},
        },
        "Inter-Rater",
    )
    rater_labels = [m[0] for m in st.metrics]
    assert "Mean dimension κ" in rater_labels
    assert "Best dimension κ" in rater_labels

    st2 = _FakeSt()
    ia._render_gold_vs_auto_block(
        st2,
        {
            "overall": {
                "n": 63,
                "cohens_kappa": 0.655,
                "agreement_rate": 0.746,
                "weighted_kappa": 0.3118,
                "kappa_ci": {"ci_lower": 0.52, "ci_upper": 0.78, "n_bootstrap": 1000},
            },
            "per_dimension": {
                "safety": {
                    "n": 16,
                    "cohens_kappa": 0.0345,
                    "agreement_rate": 0.5625,
                    "kappa_ci": {"ci_lower": 0.0, "ci_upper": 0.5},
                },
            },
        },
        "Gold",
    )
    gva_labels = [m[0] for m in st2.metrics]
    assert "Cohen's κ" in gva_labels
    assert "Agreement" in gva_labels
    assert "n" in gva_labels
    assert "Weighted κ" in gva_labels


def test_dashboard_warns_when_no_sealed_report(tmp_path):
    """With no sealed report the block degrades to a clear 'pending' notice."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import app.tabs.inter_annotator as ia

    missing = tmp_path / "part1_agreement_report.json"
    st = _FakeSt()
    with mock.patch.object(ia, "SEALED_REPORT_PATH", missing):
        ia.render_blinded_block(st)

    # Nothing to report — the 'pending' notice (warning branch) is shown instead.
    assert st.metrics == []
