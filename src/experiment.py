"""experiment.py — experiment-id helpers for the sealed human-validation run.

Centralises two pieces of logic that previously lived (hard-coded) in several
scripts and in the Makefile:

1. ``generate_experiment_id(now)`` — build the canonical experiment id from the
   run's UTC date/time, e.g. ``2026-10-15T18:30:42Z`` →
   ``trustworthiness-validation-2026-10-15-183042``. Used automatically by the
   sealer when the caller does not pin an explicit id, so a fresh run always
   produces a *new, unique, human-readable* experiment.

2. ``find_latest_experiment_id(sealed_dir)`` — read the person's current sealed
   experiment id from ``sealed_dir/manifests/sealing_manifest.json`` (there is
   exactly one canonical manifest). Consumers such as ``experiment_status.py``
   and ``validate_rerun_pipeline.py`` use this as their default, so they always
   operate on the experiment that was actually last sealed — no more hard-coded
   ``2025-08-redo`` fallback that could silently point at the wrong data.

The id is a *label*, not a cryptographic value: reproducibility comes from the
deterministic seed used to shuffle the response pool (the sealer hashes
``f"{experiment_id}:{seed}"``). Passing an explicit ``--experiment-id`` still
overrides the date-based default for fully reproducible re-seals.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The canonical manifest that records the current sealed experiment.
_SEALING_MANIFEST_NAME = "sealing_manifest.json"


def generate_experiment_id(now: Optional[datetime] = None) -> str:
    """Build an experiment id from a run's UTC timestamp.

    Format: ``trustworthiness-validation-YYYY-MM-DD-HHMMSS`` (UTC). Using the
    full timestamp guarantees uniqueness across runs on the same day while
    staying human-readable and lexicographically sortable, so a later run
    always sorts after an earlier one.

    Args:
        now: datetime to derive the id from (UTC assumed if naive). Defaults to
            the current UTC time.

    Returns:
        The experiment id string, e.g. ``trustworthiness-validation-2026-10-15-183042``.
    """
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    utc = t.astimezone(timezone.utc)
    return (
        f"trustworthiness-validation-"
        f"{utc:%Y-%m-%d-%H%M%S}"
    )


def find_latest_experiment_id(
    sealed_dir: Path | str = "experiment/sealed",
) -> Optional[str]:
    """Return the experiment id of the current sealed experiment.

    Looks up ``sealed_dir/manifests/sealing_manifest.json`` and returns its
    ``experiment_id`` field. Returns ``None`` (rather than raising) when the
    manifest is absent or malformed, so callers can fall back to a sensible
    default instead of crashing.

    Args:
        sealed_dir: the sealed-experiment root (contains ``manifests/``).

    Returns:
        The latest sealed experiment id, or ``None`` if none can be found.
    """
    manifest_path = (
        Path(sealed_dir) / "manifests" / _SEALING_MANIFEST_NAME
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest.get("experiment_id")
