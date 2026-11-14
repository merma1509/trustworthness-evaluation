#!/usr/bin/env python3
"""Move ``metadata.display_model`` out of rater-visible annotation files.

A consistency display is model-scoped, and the seal records WHICH model backs it
under ``metadata.display_model`` so an auditor can prove scoping later. That is
correct — but the field was written into the very files handed to raters, so all
22 held-out / 42 calibration consistency records exposed the model name
(``gemma3:4b`` / ``llama3.1:8b``) to an annotator who was supposed to be blind
to it. Model identity is a plausible cue for perceived truthfulness, so this was
a real blinding breach, not a cosmetic one.

This script separates the two needs:

* the rater-visible annotation files drop ``display_model`` entirely;
* a sidecar ``experiment/audit/display_model_map.jsonl`` keeps
  ``internal_key -> display_model`` so the audit trail survives.

Only ``metadata.display_model`` is removed. The usual fail-closed guard applies:
if anything else would change, nothing is written.

Usage:
    python scripts/detach_display_model.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ANNOTATION_DIR = Path("experiment/annotations")
SIDECAR = Path("experiment/audit/display_model_map.jsonl")
FIELD = "display_model"


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rater_files() -> List[Path]:
    return sorted(ANNOTATION_DIR.glob("*_RATER_*.jsonl"))


def _strip_field(records: List[dict], field: str) -> List[dict]:
    out = copy.deepcopy(records)
    for rec in out:
        meta = rec.get("metadata")
        if isinstance(meta, dict):
            meta.pop(field, None)
    return out


def detach(files: List[Path], *, dry_run: bool) -> Tuple[int, int, List[str]]:
    """Remove ``FIELD`` from each file. Returns (removed, files_ok, warnings)."""
    mapping: Dict[str, str] = {}
    removed = files_ok = 0
    warnings: List[str] = []

    for path in files:
        current = _read_jsonl(path)
        updated = copy.deepcopy(current)

        for rec in updated:
            meta = rec.get("metadata")
            if not isinstance(meta, dict) or FIELD not in meta:
                continue
            model = meta.pop(FIELD)
            key = rec.get("internal_key", "")
            if model:
                # First writer wins; a key appears once per rater file.
                mapping.setdefault(key, model)
            removed += 1

        # Fail closed: nothing but the removed field may differ.
        if _strip_field(current, FIELD) != _strip_field(updated, FIELD):
            warnings.append(f"ABORT {path.name}: changed outside {FIELD}")
            continue

        if not dry_run:
            payload = "\n".join(
                json.dumps(rec, ensure_ascii=False, sort_keys=True) for rec in updated
            )
            path.write_text(payload + "\n")
        files_ok += 1

    if warnings:
        raise SystemExit("\n".join(warnings))

    if not dry_run and mapping:
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps({"internal_key": key, FIELD: model}, sort_keys=True)
            for key, model in sorted(mapping.items())
        ]
        SIDECAR.write_text("\n".join(rows) + "\n")

    return removed, files_ok, warnings


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    files = _rater_files()
    if not files:
        print("no rater annotation files found", file=sys.stderr)
        return 1

    removed, files_ok, warnings = detach(files, dry_run=args.dry_run)

    for warning in warnings:
        print(f"  ! {warning}")
    mode = "DRY-RUN" if args.dry_run else "WRITTEN"
    print(f"{mode}: removed {FIELD} from {removed} record(s) across {files_ok} file(s)")
    if not args.dry_run:
        print(f"sidecar written: {SIDECAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
