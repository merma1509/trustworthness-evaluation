#!/usr/bin/env python3
"""Restore rater ``metadata.notes`` that were blanked by a history rewrite.

Commit ``f142958`` ("data: re-bucket annotations to new split") rewrote every
annotation file and dropped ``metadata.notes``. A follow-up commit restored
RATER_A only, on the stated grounds that "RATER_B never carried notes". That
claim is false: ``e0029b4`` holds 147/147 RATER_B calibration notes and 48/63
held-out notes, and they are substantive per-record justifications (the only
qualitative evidence behind the gold labels).

This script recovers notes from a source git revision, keyed by
``internal_key``, and writes them back.

Safety properties (deliberately strict):

* Only ``metadata.notes`` is ever written. Labels, flags, checksums and
  timestamps are read, compared and must be byte-identical afterwards.
* The script is fail-closed: if any record differs outside ``notes``, it exits
  non-zero and leaves the file untouched.
* Records already carrying a non-empty note are never overwritten unless
  ``--overwrite`` is passed; a conflict is reported instead.

Usage:
    python scripts/restore_rater_notes.py \\
        --rater RATER_B --source-rev e0029b4 [--dry-run] [--overwrite]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXPERIMENT_ID = "trustworthiness-validation-2026-10-16-042935"
ANNOTATION_DIR = Path("experiment/annotations")
SPLITS = ("calibration", "heldout")

# Provenance fields this script owns. They are derived from ``notes`` and are
# the only metadata keys (besides ``notes``) the guard below tolerates.
PROVENANCE_KEYS = ("notes_provenance", "notes_sha256", "notes_restored_from")

# Provenance values.
PROV_RATER_ORIGINAL = "rater_original"      # the rater wrote it, untouched
PROV_RESTORED = "restored"                  # recovered from an earlier revision
PROV_ABSENT = "absent_in_source"            # never existed; recorded honestly


def _note_digest(note: str) -> str:
    """Stable fingerprint of a note, so any later silent edit is detectable."""
    return hashlib.sha256(note.encode("utf-8")).hexdigest()


def _annotation_path(rater: str, split: str) -> Path:
    return ANNOTATION_DIR / f"{EXPERIMENT_ID}_{rater}_{split}.jsonl"


def _read_jsonl(text: str) -> List[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_working(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f"missing annotation file: {path}")
    return _read_jsonl(path.read_text())


def _load_revision(rev: str, path: Path) -> Optional[List[dict]]:
    """Load ``path`` as it existed at git revision ``rev``."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return _read_jsonl(proc.stdout)


def _strip_notes(records: List[dict]) -> List[dict]:
    """Deep copy with note fields removed, for equality comparison.

    Removes ``notes`` plus the derived provenance keys, so the guard can assert
    "nothing changed except the fields this script is allowed to touch".
    """
    out = copy.deepcopy(records)
    for rec in out:
        meta = rec.get("metadata")
        if isinstance(meta, dict):
            meta.pop("notes", None)
            for key in PROVENANCE_KEYS:
                meta.pop(key, None)
    return out


def _diff_outside_notes(before: List[dict], after: List[dict]) -> List[Tuple[str, str]]:
    """Return (key, reason) for every record that changed outside note fields."""
    problems: List[Tuple[str, str]] = []
    if len(before) != len(after):
        problems.append(("<file>", f"record count {len(before)} -> {len(after)}"))
        return problems
    for b, a in zip(_strip_notes(before), _strip_notes(after)):
        key = b.get("internal_key", "<no-key>")
        if b != a:
            problems.append((key, "record changed outside notes/provenance"))
    return problems


def _load_revision_notes(rater: str, revs: List[str]) -> Dict[str, str]:
    """Collect ``internal_key -> note`` across ``revs`` and *both* source splits.

    Two things make a single-file lookup wrong:

    1. Records were re-bucketed between calibration and held-out, so a key that
       now lives in held-out may have carried its note in the calibration file
       at the source revision.
    2. Older annotations only exist in earlier revisions (``e0029b4`` predates
       15 keys; ``1a0a2ae`` carries 14 of them).

    Revisions are tried in the given order and the first non-empty note wins,
    so pass the most authoritative revision first.
    """
    notes: Dict[str, str] = {}
    for rev in revs:
        for split in SPLITS:
            source = _load_revision(rev, _annotation_path(rater, split))
            if source is None:
                continue
            for rec in source:
                note = (rec.get("metadata") or {}).get("notes") or ""
                key = rec.get("internal_key", "")
                if note.strip() and key not in notes:
                    notes[key] = note
    return notes


def restore_split(
    rater: str,
    split: str,
    source_notes: Dict[str, str],
    source_revs: List[str],
    *,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[int, int, int, int, List[str]]:
    """Restore one split. Returns (filled, kept, absent, unmatched, warnings).

    Every record ends with an explicit provenance stamp, so a reader can tell a
    rater-written note from a restored one, and from one that never existed.
    The absent case is recorded rather than filled in: inventing a note in the
    rater's voice would make fabricated evidence indistinguishable from real.
    """
    path = _annotation_path(rater, split)
    current = _load_working(path)
    warnings: List[str] = []

    filled = kept = absent = unmatched = 0
    updated: List[dict] = []

    for rec in current:
        rec = copy.deepcopy(rec)
        key = rec.get("internal_key", "")
        rec.setdefault("metadata", {})
        meta = rec["metadata"]
        existing = meta.get("notes") or ""

        if key in source_notes:
            note = source_notes[key]
            if existing.strip() and not overwrite:
                # A note is already present; keep it and report if it differs.
                kept += 1
                meta["notes"] = existing
                meta["notes_provenance"] = PROV_RATER_ORIGINAL
                if existing != note:
                    warnings.append(f"{key}: kept existing note (differs from source)")
            else:
                filled += 1
                meta["notes"] = note
                meta["notes_provenance"] = PROV_RESTORED
            meta["notes_sha256"] = _note_digest(meta["notes"])
            meta["notes_restored_from"] = " ".join(source_revs)
        elif existing.strip():
            # Not in any source revision, but the record already carries one.
            kept += 1
            meta["notes"] = existing
            meta["notes_provenance"] = PROV_RATER_ORIGINAL
            meta["notes_sha256"] = _note_digest(existing)
        else:
            # No source note ever existed: record that honestly, do not invent.
            absent += 1
            unmatched += 1
            meta["notes"] = ""
            meta["notes_provenance"] = PROV_ABSENT
            meta.pop("notes_sha256", None)
            warnings.append(f"{key}: no source note exists; marked {PROV_ABSENT}")

        updated.append(rec)

    # Fail closed: nothing but note fields may differ.
    problems = _diff_outside_notes(current, updated)
    if problems:
        for key, reason in problems[:10]:
            warnings.append(f"ABORT {key}: {reason}")
        raise SystemExit(
            f"refusing to write {path}: {len(problems)} record(s) changed outside notes"
        )

    if not dry_run:
        payload = "\n".join(
            json.dumps(rec, ensure_ascii=False, sort_keys=True) for rec in updated
        )
        path.write_text(payload + "\n")

    return filled, kept, absent, unmatched, warnings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater", required=True, help="e.g. RATER_B")
    parser.add_argument(
        "--source-rev",
        required=True,
        nargs="+",
        help="git revision(s) to recover notes from, e.g. e0029b4 1a0a2ae "
        "(tried in order; first non-empty note wins)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    grand_filled = grand_kept = grand_absent = grand_unmatched = 0
    all_warnings: List[str] = []

    # Index the source revision across BOTH splits: re-bucketing moved keys
    # between calibration and held-out, so a per-file lookup would miss most.
    source_notes = _load_revision_notes(args.rater, args.source_rev)
    print(f"source notes at {' '.join(args.source_rev)}: {len(source_notes)} keys")

    for split in SPLITS:
        filled, kept, absent, unmatched, warnings = restore_split(
            args.rater,
            split,
            source_notes,
            args.source_rev,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        grand_filled += filled
        grand_kept += kept
        grand_absent += absent
        grand_unmatched += unmatched
        all_warnings.extend(warnings)
        print(
            f"{args.rater} {split:12s} filled={filled:3d} kept={kept:3d} "
            f"absent={absent:2d}"
        )

    for warning in all_warnings:
        print(f"  ! {warning}")

    mode = "DRY-RUN" if args.dry_run else "WRITTEN"
    print(
        f"\n{mode}: {grand_filled} note(s) restored, {grand_kept} kept, "
        f"{grand_absent} marked {PROV_ABSENT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

