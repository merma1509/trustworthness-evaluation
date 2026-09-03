#!/usr/bin/env python3
"""verify_immutable.py — verify artifacts match their committed checksums.

Implements the artifact-integrity.json file in the results directory. Reads a manifest
(e.g. ``results/manifest.json``) whose ``checksums`` section maps relative
artifact paths to a SHA-256 prefix, and reports any MISSING / MODIFIED file.

This is intentionally schema-agnostic over the manifest's exact shape: it
accepts either
  { "checksums": { "rel/path.json": "sha256:abcd1234..." , ... } }
or a flat
  { "rel/path.json": "sha256:abcd1234...", ... }

Usage:
    python3 scripts/verify_immutable.py \
        --manifest results/manifest.json \
        --base-dir .

Exits 0 if all artifacts exist and match, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sealing import sha256_file  # noqa: E402


def _extract_checksums(manifest: Dict) -> Dict[str, str]:
    """Return ``{relative_path: sha256_hex_prefix}`` from a manifest doc.

    Accepts both a nested ``{"checksums": {...}}`` layout and a flat layout.
    """
    if "checksums" in manifest and isinstance(manifest["checksums"], dict):
        return dict(manifest["checksums"])
    return {
        k: v
        for k, v in manifest.items()
        if isinstance(v, str) and v.startswith("sha256:")
    }


def verify_immutable_artifacts(
    base_dir: Path,
    manifest_path: Path,
) -> bool:
    """Verify every artifact listed in the manifest has not been modified.

    Args:
        base_dir: Directory the manifest's relative paths are anchored to.
        manifest_path: Path to the manifest JSON file.

    Returns:
        True if every artifact exists and its SHA-256 prefix matches.
    """
    if not manifest_path.exists():
        print(f"  ✗ Manifest not found: {manifest_path}")
        return False

    manifest = json.loads(manifest_path.read_text())
    checksums = _extract_checksums(manifest)

    if not checksums:
        print("  ✗ Manifest has no 'checksums' section / no sha256: entries.")
        return False

    failures: List[str] = []

    for artifact_path, expected_hash in sorted(checksums.items()):
        full_path = base_dir / artifact_path
        # Strip any leading "sha256:" prefix from the value.
        expected = expected_hash.split(":", 1)[1] if ":" in expected_hash else expected_hash

        if not full_path.exists():
            failures.append(f"MISSING: {artifact_path}")
            continue

        actual = sha256_file(full_path)
        if actual[:16] != expected[:16]:
            failures.append(
                f"MODIFIED: {artifact_path} "
                f"(expected sha256:{expected[:16]}, got sha256:{actual[:16]})"
            )

    if failures:
        print("  VERIFICATION FAILED:")
        for f in failures:
            print(f"    [FAIL] {f}")
        return False

    n = len(checksums)
    print(f"  ✓ {n} artifact(s) verified. No modifications detected.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to the checksum manifest JSON.")
    parser.add_argument("--base-dir", default=".",
                        help="Root directory for manifest-relative paths (default: .).")
    args = parser.parse_args()

    ok = verify_immutable_artifacts(
        Path(args.base_dir), Path(args.manifest)
    )
    print("\n  RESULT:", "PASS — artifacts immutable." if ok else "FAIL.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
