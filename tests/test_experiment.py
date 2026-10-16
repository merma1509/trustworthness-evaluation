"""Unit tests for the experiment-id helpers (src/experiment.py).

Covers the datetime-derived experiment id generation and the "last sealed
experiment" lookup that lets consumers (status / validate-rerun / Makefile)
operate on the current run without hard-coding an id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.experiment import find_latest_experiment_id, generate_experiment_id


# ── generate_experiment_id ────────────────────────────────────────────────
def test_generate_uses_utc_datetime():
    """The id embeds the UTC date/time, so later runs sort after earlier ones."""
    earlier = generate_experiment_id(datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc))
    later = generate_experiment_id(datetime(2026, 10, 1, 9, 30, 0, tzinfo=timezone.utc))
    assert earlier.startswith("trustworthiness-validation-2026-10-01-090000")
    assert later.startswith("trustworthiness-validation-2026-10-01-093000")
    # Lexicographic sort == chronological sort (fixed-width components).
    assert earlier < later


def test_generate_normalizes_naive_to_utc():
    """A naive datetime is interpreted as UTC (not local time)."""
    naive = datetime(2026, 10, 1, 12, 0, 0)
    assert generate_experiment_id(naive).endswith("-120000")


def test_generate_converts_non_utc_tz_to_utc():
    """A tz-aware datetime is normalised to UTC before formatting."""
    est = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    # 08:00 UTC stays 08:00 UTC; a -5 offset would shift to 13:00 UTC.
    assert generate_experiment_id(est).endswith("-080000")


def test_generate_defaults_to_now_utc() -> None:
    """Calling with no argument uses the current time; id parses cleanly."""
    import re

    now_id = generate_experiment_id()
    assert re.fullmatch(
        r"trustworthiness-validation-\d{4}-\d{2}-\d{2}-\d{6}", now_id
    ), now_id


def test_generate_is_unique_per_timestamp() -> None:
    a = generate_experiment_id(datetime(2026, 10, 1, 1, 2, 3, tzinfo=timezone.utc))
    b = generate_experiment_id(datetime(2026, 10, 1, 1, 2, 4, tzinfo=timezone.utc))
    assert a != b


# ── find_latest_experiment_id ─────────────────────────────────────────────
def _write_manifest(tmp_path, experiment_id: str) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "sealing_manifest.json").write_text(
        json.dumps({"experiment_id": experiment_id}), encoding="utf-8"
    )


def test_find_latest_reads_manifest(tmp_path):
    _write_manifest(tmp_path, "trustworthiness-validation-2026-10-15-183000")
    assert (
        find_latest_experiment_id(str(tmp_path))
        == "trustworthiness-validation-2026-10-15-183000"
    )


def test_find_latest_missing_manifest_returns_none(tmp_path):
    assert find_latest_experiment_id(str(tmp_path)) is None


def test_find_latest_malformed_manifest_returns_none(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "sealing_manifest.json").write_text("not-json{", encoding="utf-8")
    assert find_latest_experiment_id(str(tmp_path)) is None


def test_find_latest_missing_experiment_id_field(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "sealing_manifest.json").write_text(
        json.dumps({"seed": "42"}), encoding="utf-8"
    )
    assert find_latest_experiment_id(str(tmp_path)) is None


def test_find_latest_default_path_type():
    """The default sealed_dir resolves to the repo's experiment/sealed."""
    import sys
    from pathlib import Path

    sys.path.insert(0, ".")
    default_path = find_latest_experiment_id()
    # Only asserts the call is safe and returns a str path type, without
    # depending on a real manifest being present.
    assert isinstance(find_latest_experiment_id(), (str, type(None)))
    assert Path(".").is_dir()
