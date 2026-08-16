"""Tests that the code constants match the canonical scoring spec (Task 8).

This guards against silent drift between docs/scoring-spec-v2.md and the
Python thresholds / weights / group counts actually used by the pipeline.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC = PROJECT_ROOT / "docs" / "scoring-spec-v2.md"


def _spec_exists():
    return SPEC.exists()


def test_spec_file_exists():
    assert SPEC.exists(), "docs/scoring-spec-v2.md missing"


def test_sim_threshold_matches_code():
    """The spec & code must agree on the consistency similarity threshold."""
    from src.consistency import DEFAULT_SIMILARITY_THRESHOLD
    spec_text = SPEC.read_text()
    assert "0.85" in spec_text
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.85
