"""Audit-trail tests for rater ``metadata.notes``.

Notes are the only qualitative evidence behind the gold labels. They have been
lost twice in this repository's history:

* ``f142958`` rewrote every annotation file and blanked all notes;
* the follow-up repair restored RATER_A only, asserting (incorrectly) that
  "RATER_B never carried notes" — 195 RATER_B notes were in fact available.

These tests pin the invariants that make a silent loss detectable:

1. every annotated record carries a stamped provenance value;
2. a non-empty note has a matching ``notes_sha256`` (so an edit is visible);
3. the restore script refuses to touch anything but ``notes``/provenance;
4. an unrecoverable note is recorded as ``absent_in_source``, never invented.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from restore_rater_notes import (  # noqa: E402
    PROV_ABSENT,
    PROV_RATER_ORIGINAL,
    PROV_RESTORED,
    PROVENANCE_KEYS,
    SPLITS,
    _annotation_path,
    _diff_outside_notes,
    _note_digest,
)

ANNOTATION_GLOB = "*_RATER_*.jsonl"
VALID_PROVENANCE = {PROV_RATER_ORIGINAL, PROV_RESTORED, PROV_ABSENT}


def _rater_files() -> list[Path]:
    return sorted((REPO_ROOT / "experiment/annotations").glob(ANNOTATION_GLOB))


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rater_files_exist() -> None:
    """Guard against the glob silently matching nothing."""
    assert _rater_files(), "no *_RATER_*.jsonl annotation files found"


@pytest.mark.parametrize("path", _rater_files(), ids=lambda p: p.name)
def test_every_record_has_stamped_provenance(path: Path) -> None:
    """Every record must declare where its note came from."""
    for rec in _records(path):
        meta = rec.get("metadata") or {}
        provenance = meta.get("notes_provenance")
        assert provenance in VALID_PROVENANCE, (
            f"{path.name}:{rec.get('internal_key')} has provenance "
            f"{provenance!r}; expected one of {sorted(VALID_PROVENANCE)}"
        )


@pytest.mark.parametrize("path", _rater_files(), ids=lambda p: p.name)
def test_note_digest_matches_note(path: Path) -> None:
    """A non-empty note must be fingerprinted; an empty one must not be.

    This is what makes a later silent edit detectable: change the text and the
    digest no longer matches.
    """
    for rec in _records(path):
        meta = rec.get("metadata") or {}
        note = meta.get("notes") or ""
        digest = meta.get("notes_sha256")

        if note.strip():
            assert digest, (
                f"{path.name}:{rec.get('internal_key')} has a note but no "
                "notes_sha256 — the audit trail is unverifiable"
            )
            expected = hashlib.sha256(note.encode("utf-8")).hexdigest()
            assert digest == expected, (
                f"{path.name}:{rec.get('internal_key')} note does not match its "
                "digest: the note was edited after the stamp was written"
            )
        else:
            assert not digest, (
                f"{path.name}:{rec.get('internal_key')} is empty but carries a "
                "digest; provenance is inconsistent"
            )


@pytest.mark.parametrize("path", _rater_files(), ids=lambda p: p.name)
def test_absent_notes_are_marked_not_invented(path: Path) -> None:
    """An ``absent_in_source`` record must be empty and carry no digest."""
    for rec in _records(path):
        meta = rec.get("metadata") or {}
        if meta.get("notes_provenance") != PROV_ABSENT:
            continue
        assert not (meta.get("notes") or "").strip(), (
            f"{path.name}:{rec.get('internal_key')} is marked absent_in_source "
            "but carries text — an invented note would be indistinguishable "
            "from a rater's own"
        )
        assert "notes_sha256" not in meta


def test_restore_guard_rejects_non_note_changes() -> None:
    """The restore script must refuse to write anything but note fields."""
    base = {
        "internal_key": "K_0001",
        "annotations": {"safety_label": "REFUSE"},
        "flags": {"rater_uncertainty": "confident"},
        "metadata": {"notes": "original", "template_checksum": "abc"},
    }
    # Identical apart from note fields -> allowed.
    touched_notes = json.loads(json.dumps(base))
    touched_notes["metadata"]["notes"] = "restored"
    touched_notes["metadata"]["notes_provenance"] = PROV_RESTORED
    assert _diff_outside_notes([base], [touched_notes]) == []

    # A changed label must be reported.
    changed_label = json.loads(json.dumps(base))
    changed_label["annotations"]["safety_label"] = "COMPLY"
    problems = _diff_outside_notes([base], [changed_label])
    assert problems, "guard failed to catch a changed label"

    # A changed checksum must be reported.
    changed_checksum = json.loads(json.dumps(base))
    changed_checksum["metadata"]["template_checksum"] = "xyz"
    assert _diff_outside_notes([base], [changed_checksum])

    # A dropped record must be reported.
    assert _diff_outside_notes([base], [])


def test_note_digest_is_stable_and_sensitive() -> None:
    """The digest must be deterministic and change on any edit."""
    note = "Model firmly declines the explicit content request"
    assert _note_digest(note) == _note_digest(note)
    assert _note_digest(note) != _note_digest(note + ".")
    assert len(_note_digest(note)) == 64


def test_notes_are_not_lost_relative_to_head() -> None:
    """Notes must not silently disappear again in a future commit.

    Compares the working tree against ``HEAD``; skipped when the files are not
    tracked yet or no git history is available.
    """
    for split in SPLITS:
        path = _annotation_path("RATER_B", split)
        rel = path.as_posix()
        proc = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            pytest.skip(f"{rel} has no HEAD revision")
        head_records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        head_notes = sum(
            1 for r in head_records if ((r.get("metadata") or {}).get("notes") or "").strip()
        )
        working_notes = sum(
            1 for r in _records(path) if ((r.get("metadata") or {}).get("notes") or "").strip()
        )
        assert working_notes >= head_notes, (
            f"{rel}: working tree has {working_notes} notes but HEAD has "
            f"{head_notes} — notes were dropped"
        )
