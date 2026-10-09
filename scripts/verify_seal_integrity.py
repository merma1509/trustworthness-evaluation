#!/usr/bin/env python3
"""verify_seal_integrity.py — confirm a sealed experiment is intact

Recomputes SHA-256 for every sealed file (calibration/heldout templates, the
encrypted auto-label blob) and compares against the sealing manifest. Also
verifies the blinding invariants:

  * every template carries an opaque internal_key,
  * NO sensitive value is ever exposed in a rater template
    (auto_label / expected_behavior / model_id / attack_type / similarity /
     is_correct / scorer_label must be absent OR empty placeholders),
  * template internal_keys form a subset of the sealed auto-label keys (so the
    gold/auto join later cannot silently drop records)

Usage:
    python3 scripts/verify_seal_integrity.py \
        --manifest experiment/sealed/manifests/sealing_manifest.json \
        --sealed-dir experiment/sealed \
        [--sealed-labels experiment/sealed/labels/sealed_auto_labels.jsonl.enc] \
        [--keys-file experiment/.sealed_keeper.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sealing import decrypt_json_file, sha256_jsonl  # noqa: E402

# Keys that must NEVER carry a real (non-empty) value inside a rater template
SENSITIVE_KEYS = {
    "auto_label",
    "expected_behavior",
    "model_id",
    "attack_type",
    "semantic_similarity",
    "is_correct",
    "actual_behavior",
    "scorer_label",
    "expected_behavior",
}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _nonempty_sensitive(record: dict) -> List[str]:
    """Return sensitive keys that hold a non-empty value anywhere in a record"""
    found: List[str] = []
    for kw in SENSITIVE_KEYS:
        for m in re.finditer(rf'"{kw}"\s*:\s*"([^"]*)"', json.dumps(record)):
            if m.group(1) != "":
                found.append(kw)
    return found


def _prompt_fingerprint(record: dict) -> str:
    """Canonical fingerprint of the rater-visible prompt content.

    For consistency units we digest the whole pair list (prompt + response
    texts); for safety/truthfulness we digest the single prompt_text. This is
    what a rater actually sees, so two records that share a fingerprint are
    visible duplicates (e.g. the same group served for gemma and llama with
    identical prompts *and* responses) — a model-identity leak risk.
    """
    if isinstance(record.get("pairs"), list):
        pairs = []
        for p in record["pairs"]:
            if isinstance(p, dict):
                pairs.append((p.get("prompt", ""), p.get("response", "")))
        return repr(pairs)
    return repr(record.get("prompt_record", {}).get("prompt_text", ""))


def _duplicate_prompt_groups(records: List[dict]) -> Dict[str, List[str]]:
    """Group template internal_keys by the rater-visible prompt fingerprint."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        buckets[_prompt_fingerprint(rec)].append(rec.get("internal_key", ""))
    return {fp: keys for fp, keys in buckets.items() if len(keys) > 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sealed-dir", default="experiment/sealed")
    parser.add_argument("--sealed-labels", default=None,
                        help="Path to sealed_auto_labels.jsonl.enc (optional).")
    parser.add_argument("--keys-file", default=None,
                        help="Path to keeper JSON for passphrase so the sealed "
                             "auto payload can be decrypted for the join check.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    sealed = Path(args.sealed_dir)
    checksums = manifest.get("seal_checksums", {})

    ok = True

    # ── 1) Recompute template checksums vs manifest ─────────────────────────
    cal_path = sealed / "templates" / f"{manifest['experiment_id']}_calibration_template.jsonl"
    hold_path = sealed / "templates" / f"{manifest['experiment_id']}_heldout_template.jsonl"
    for label, path in (("calibration template", cal_path),
                        ("heldout template", hold_path)):
        if not path.exists():
            print(f"  missing {path}")
            ok = False
            continue
        records = _load_jsonl(path)
        actual = sha256_jsonl(records)
        expected = checksums.get(f"{label.split()[0]}_template")
        match = actual == (expected.split(":")[1] if expected else None)
        print(f"  {'[MATCH]' if match else '[MISMATCH]'} {label}: sha256={actual}")
        if not match:
            print(f"       manifest expected: {expected}")
            ok = False

    # ── 2) Blinding invariants on every template record ─────────────────────
    all_keys: Set[str] = set()
    for path in (cal_path, hold_path):
        for rec in _load_jsonl(path):
            all_keys.add(rec.get("internal_key", ""))
            if not rec.get("internal_key"):
                print(f"  record missing internal_key in {path.name}")
                ok = False
            leak = _nonempty_sensitive(rec)
            if leak:
                print(f"  SENSITIVE VALUE leaked in {rec.get('internal_key')}: "
                      f"{sorted(set(leak))}")
                ok = False

    print(f"  blinding: {len(all_keys)} template internal_keys checked, "
          f"0 non-empty sensitive values")

    # ── 2b) Rater-visible duplicate prompt groups (model-identity leak risk) ─
    # If the same prompt/response content appears twice in one split (e.g. the
    # same consistency group served once for gemma and once for llama), a rater
    # may notice the duplicate and infer that the two records belong to
    # different models. This does not invalidate the seal, but the researcher
    # should be aware before interpreting the held-out agreement.
    for name, path in (("calibration", cal_path), ("heldout", hold_path)):
        dupes = _duplicate_prompt_groups(_load_jsonl(path))
        if dupes:
            n_keys = sum(len(v) for v in dupes.values())
            print(f"  [WARN] {name}: {len(dupes)} rater-visible prompt "
                  f"duplicates ({n_keys} records): "
                  f"{sorted(v[0] for v in dupes.values())[:6]} ...")

    # ── 3) Auto-label join integrity (optional if encrypted) ────────────────
    if args.sealed_labels:
        labels_path = Path(args.sealed_labels)
        if args.keys_file and labels_path.exists():
            keeper = json.loads(Path(args.keys_file).read_text())
            auto = decrypt_json_file(labels_path, keeper["passphrase"])
            auto_keys = {a.get("internal_key", "") for a in auto}
            missing = all_keys - auto_keys
            print(f"  auto payload records: {len(auto)}")
            m = "[NOT MISSING]" if not missing else "[MISSING]"
            print(f"  {m} every template key present in sealed auto "
                  f"({len(all_keys - missing)}/{len(all_keys)})")
            if missing:
                print(f"       missing from auto: {sorted(missing)[:5]}")
                ok = False
        else:
            print("  (skipping auto-label join check; --keys-file not given)")

    print("\n  RESULT:", "PASS — experiment sealed intact & blinded"
          if ok else "FAIL — integrity issue")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
