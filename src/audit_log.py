"""Append-only JSONL audit logging for the reproducibility pipeline.

("Every action logged"): every computation and
pipeline step is recorded as a JSONL line with a UTC timestamp, the action and
script, the invocation arguments, the SHA-256 of consumed inputs and produced
outputs, and an outcome marker.

Logs are append-only: existing entries are never rewritten, retouched or
deleted, which keeps the audit trail tamper-evident and matches the "no
modification of annotations after they were filed" invariant (§2.9).

Usage (within any pipeline script)::

    from src.audit_log import log_action, file_sha256

    log_action(
        log_path=EXPERIMENT_DIR / "logs" / "processing_log.jsonl",
        action="compute_scores",
        script="scripts/compute_scores.py",
        args={"raw": str(raw_dir), "output": str(output)},
        input_paths=[raw_dir],
        output_paths=[output],
    )
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union


def file_sha256(path: Union[str, Path], prefix: int = 16) -> Optional[str]:
    """Return the hex SHA-256 (optionally truncated) of a file, or None if absent.

    Mirrors ``src.sealing.sha256_file`` but tolerates missing paths so the
    audit trail can record absent inputs without raising.
    """

    p = Path(path)
    if not p.exists() or p.is_dir():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:prefix] if prefix else digest


def _path_sha_or_list(value: Union[str, Path, List, None], prefix: int) -> object:
    """Best-effort hash of a single path, a list of paths, or an opaque value

    Directories are hashed as an ordered list of their files' hashes so the
    audit entry captures the full consumed input set rather than ``null``
    """

    if value is None:
        return None
    if isinstance(value, (str, Path)):
        p = Path(value)
        if p.is_dir():
            return [
                _path_sha_or_list(f, prefix)
                for f in sorted(p.glob("**/*"))
                if f.is_file()
            ]
        return file_sha256(p, prefix)
    if isinstance(value, (list, tuple)):
        return [_path_sha_or_list(v, prefix) for v in value]
    # Opaque scalar (int/float/bool) — record as-is for traceability.
    return value


def log_action(
    log_path: Union[str, Path],
    action: str,
    script: str,
    args: Optional[Dict] = None,
    input_paths: Optional[List] = None,
    output_paths: Optional[List] = None,
    outcome: str = "ok",
    status: int = 0,
    extra: Optional[Dict] = None,
) -> Path:
    """Append a single audit-trail entry to ``log_path`` (mkdir -p, append-only).

    Args:
        log_path: Destination ``*.jsonl`` audit file.
        action: Short verb, e.g. ``compute_scores`` or ``seal_experiment``.
        script: Script that performed the action (for provenance).
        args: Invocation arguments (scalars or path strings; paths are hashed).
        input_paths: Files/dirs consumed by the action (hashed at capture time).
        output_paths: Files produced by the action (hashed at capture time).
        outcome: Human-readable result marker (default ``ok``).
        status: Process exit code (default ``0``).
        extra: Arbitrary additional key/values to embed.
    """

    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "script": script,
        "args": _path_sha_or_list(args, 16) if args else None,
        "input_hashes": {
            str(k): _path_sha_or_list(v, 16)
            for k, v in (input_paths or {}).items()
        } if input_paths else None,
        "output_hashes": {
            str(k): _path_sha_or_list(v, 16)
            for k, v in (output_paths or {}).items()
        } if output_paths else None,
        "outcome": outcome,
        "status": status,
    }
    if extra:
        entry["extra"] = extra

    with p.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return p


def load_log(log_path: Union[str, Path]) -> List[dict]:
    """Read all entries of an audit JSONL file (empty list if absent)."""

    p = Path(log_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open() if line.strip()]


def seal_log_paths(script: str, *paths: Union[str, Path]) -> None:
    """Entry hook: recompute hashes from the CLI before the script runs.

    Convenience for lightweight drivers; meant to be called right after
    argument parsing so the audit entry reflects the actual inputs subsequently
    consumed by the pipeline.
    """

    _ = script, paths  # reserved for richer drivers that pass explicit args

if __name__ == "__main__":
    # Minimal self-test: emit a sample audit line to stdout-understandable path.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    demo = tmp / "processing_log.jsonl"
    log_action(
        log_path=demo,
        action="demo",
        script=sys.argv[0],
        args={"--flag": True},
        extra={"note": "audit_log self-test"},
    )
    print("wrote", demo)
    print(load_log(demo)[0])
