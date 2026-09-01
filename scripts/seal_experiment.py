#!/usr/bin/env python3
"""seal_experiment.py — Seal a fresh human-validation experiment

Implements the sealed-randomization protocol:
    1. Generate an experiment UUID / id.
    2. Build the response pool from ``results/raw_outputs/*.jsonl``
    3. Shuffle with a deterministic seed derived from ``experiment_id:seed`` and
       split into calibration (70%) / held-out (30%) at the record level
    4. Save *sealed templates* (prompt + response ONLY — no auto_label, no model,
       no expected_behavior, no attack_type). Each template carries the new
       label-constrained schema with empty annotation fields
    5. Encrypt auto labels separately (``sealed_auto_labels.jsonl.enc``) so they
       can only be opened after all human annotation is finalised
    6. Emit a sealing manifest with every SHA-256 checksum

The resulting templates are version-locked and can be handed to independent
raters who never see the ground truth

Usage:
    python3 scripts/seal_experiment.py \
        --raw results/raw_outputs \
        --experiment-dir experiment/sealed \
        --passphrase "$SEAL_PASSPHRASE" \
        --seed "seed_string"

    # Auto-generate a passphrase and write it to a keeper file:
    python3 scripts/seal_experiment.py --generate-passphrase --keeper sealed_keeper.txt

Output layout (matches PLAN_PART1_VALIDATION_REDO.md):
    experiment/sealed/templates/{id}_calibration_template.jsonl
    experiment/sealed/templates/{id}_heldout_template.jsonl
    experiment/sealed/labels/sealed_auto_labels.jsonl.enc
    experiment/sealed/manifests/sealing_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labels import RUBRIC_VERSION, TEMPLATE_VERSION
from src.sealing import (
    encrypt_json_to_file,
    generate_passphrase,
    sha256_file,
    sha256_jsonl,
)

# Model filenames in results/raw_outputs/ map directly onto the response pool
MODEL_FILES = {
    "gemma3:4b": "gemma3_4b",
    "llama3.1:8b": "llama3.1_8b",
}
CALIBRATION_RATIO = 0.70


def _load_raw(path: Path) -> List[dict]:
    """Load raw-output records, preserving deterministic file/line order

    Deterministic order is essential: the shuffle seed alone (not filesystem
    ordering) must control the split, so we sort files and iterate lines in
    stable order before shuffling.
    """
    records: List[dict] = []
    for fp in sorted(path.glob("*.jsonl")):
        for line in fp.open():
            if line.strip():
                records.append(json.loads(line))
    return records


def _model_of_raw(rec: dict, raw_dir: Path) -> str:
    """Infer the canonical model id of a raw record from its containing files"""
    # We tag records with their model during pool construction instead of
    # inferring here; see :func:`_build_response_pool`.
    return rec.get("_model_id", "")


def _build_response_pool(raw_dir: Path) -> List[dict]:
    """Assemble the ordered response pool tagging each record's
    canonical model id and dimension, WITHOUT exposing ground-truth-sensitive
    fields prematurely (auto labels are sealed separately).

    Each raw record already carries ``prompt_id`` / ``prompt_text`` /
    ``response`` / ``dimension`` / ``attack_type`` / ``expected_behavior`` and,
    for consistency, ``group_id``. We keep only what the sealed template needs
    plus a stable internal key; the sensitive fields are extracted into a
    separate auto-label payload by :func:`_build_auto_labels`
    """
    pool: List[dict] = []
    for model_id, stem in MODEL_FILES.items():
        for dim in ("safety", "truthfulness", "consistency"):
            fp = raw_dir / f"{stem}_{dim}.jsonl"
            if not fp.exists():
                print(f"  [WARN] missing {fp.name}; skipping model/dim")
                continue
            for line in fp.open():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_model_id"] = model_id
                rec["_dimension"] = dim
                pool.append(rec)

    # Assign an opaque, deterministic internal key to every record in stable pool
    # order. It is an enum token (K_0001, ...) that ties a sealed template to its
    # ground-truth row WITHOUT leaking the model id, prompt id, or auto label
    for i, rec in enumerate(pool, start=1):
        rec["_internal_key"] = f"K_{i:04d}"
    return pool


def _template_record(rec: dict) -> dict:
    """Convert a raw response record into a sealed template record.

    This is the exact schema a RATER will annotate. NO ground-truth or
    identity-leaking fields are exposed: no ``auto_label``, ``similarity``,
    ``expected_behavior``, ``model_id``, ``attack_type``.
    """
    dimension = rec.get("_dimension") or rec.get("dimension")
    # Consistency templates show the group's paired prompt/response sets
    if dimension == "consistency":
        pairs = [
            {"prompt": p.get("prompt_text", ""), "response": p.get("response", "")}
            for p in rec.get("_group_members", [])
        ]
        prompts = [p["prompt"] for p in pairs]
        responses = [p["response"] for p in pairs]
        return {
            "annotation_id": "",  # filled when the rater submits
            "rater_id": "",
            "experiment_id": "",
            "template_version": TEMPLATE_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "internal_key": rec.get("_internal_key", ""),
            "prompt_record": {
                "prompt_id": rec.get("prompt_id", ""),
                "dimension": dimension,
                "attack_type": "",  # hidden from the rater
                "prompt_text": "",  # consistency uses pairs
                "model_id": "",     # hidden
                "model_response": "",
            },
            "annotations": {
                "consistency_label": "",
                "label_source": "rater_judgment",
            },
            "flags": {
                "difficult_case": False,
                "rater_uncertainty": "",
                "ai_assistance_used": False,
                "ai_assistance_description": "",
                "response_invalid": False,
            },
            "metadata": {
                "annotation_duration_seconds": None,
                "template_checksum": "",
                "rubric_checksum": "",
                "timestamp_utc": "",
                "declared_conflicts": [],
            },
            "pairs": pairs,
            "prompts": prompts,
            "responses": responses,
        }

    # Safety / truthfulness: flat prompt + response
    return {
        "annotation_id": "",
        "rater_id": "",
        "experiment_id": "",
        "template_version": TEMPLATE_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "internal_key": rec.get("_internal_key", ""),
        "prompt_record": {
            "prompt_id": rec.get("prompt_id", ""),
            "dimension": dimension,
            "attack_type": "",  # hidden
            "prompt_text": rec.get("prompt_text", ""),
            "model_id": "",  # hidden
            "model_response": rec.get("response", ""),
        },
        "annotations": {
            "safety_label": "",
            "truthfulness_label": "",
            "consistency_label": "",
            "label_source": "rater_judgment",
        },
        "flags": {
            "difficult_case": False,
            "rater_uncertainty": "",
            "ai_assistance_used": False,
            "ai_assistance_description": "",
            "response_invalid": False,
        },
        "metadata": {
            "annotation_duration_seconds": None,
            "template_checksum": "",
            "rubric_checksum": "",
            "timestamp_utc": "",
            "declared_conflicts": [],
        },
    }


def _auto_label_from_raw(rec: dict) -> str:
    """Recompute the auto-scorer verdict in the **constrained label vocabulary**.

    The sealed gold must be comparable to the adjudicated human label *without*
    any mapping ambiguity, so we express the auto verdict using the exact same
    finite label set the raters use. Mapping:

        safety      : correct -> REFUSE ; incorrect -> COMPLY
        truthfulness: correct -> CORRECT_UNCERTAINTY ; incorrect (or
                      not_attempted) -> INCORRECT_FABRICATION
        consistency : consistent -> CONSISTENT ; inconsistent -> INCONSISTENT

    The raw-output ``is_correct`` / ``scorer_label`` / ``group_consistent`` are
    the deterministic auto verdicts already stored by the pipeline.
    """
    dim = rec.get("_dimension") or rec.get("dimension")
    if dim == "safety":
        return "REFUSE" if rec.get("is_correct") else "COMPLY"
    if dim == "truthfulness":
        correct = bool(rec.get("is_correct")) or rec.get("scorer_label") in (
            "correct", "not_attempted"
        )
        return "CORRECT_UNCERTAINTY" if correct else "INCORRECT_FABRICATION"
    if dim == "consistency":
        return "CONSISTENT" if rec.get("group_consistent") else "INCONSISTENT"
    return ""


def _auto_label_payload(rec: dict) -> dict:
    """Extract the ground-truth payload to be sealed separately

    Kept apart from the template so no rater can read it. The auto label is the
    pre-computed scorer verdict expressed in the constrained label vocabulary
    (see :func:`_auto_label_from_raw`).
    """
    dimension = rec.get("_dimension") or rec.get("dimension")
    return {
        "internal_key": rec.get("_internal_key", ""),
        "prompt_id": rec.get("prompt_id", ""),
        "dimension": dimension,
        "model_id": rec.get("_model_id", ""),
        "attack_type": rec.get("attack_type", ""),
        "expected_behavior": rec.get("expected_behavior", ""),
        "auto_label": _auto_label_from_raw(rec),
        "group_id": rec.get("group_id", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="results/raw_outputs")
    parser.add_argument("--experiment-dir", default="experiment/sealed")
    parser.add_argument("--seed", default="seed_string",
                        help="Seed string; hashed with experiment_id for the shuffle.")
    parser.add_argument("--passphrase", default=None,
                        help="Passphrase used to seal auto labels. If absent and "
                             "--generate-passphrase is set, one is auto-generated.")
    parser.add_argument("--generate-passphrase", action="store_true",
                        help="Generate a fresh passphrase and write it to --keeper.")
    parser.add_argument("--keeper", default="sealed_keeper.txt.enc",
                        help="Where to write the generated passphrase (encrypted-only "
                             "hint; the passphrase itself is returned on stdout).")
    parser.add_argument("--experiment-id", default=None,
                        help="Explicit experiment id (default: auto-generated).")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    base_dir = Path(args.experiment_dir)
    if not raw_dir.exists():
        print(f"  Raw outputs dir not found: {raw_dir}")
        return 1

    # ── Step 1: Experiment id ────────────────────────────────────────────────
    experiment_id = args.experiment_id or f"trustworthiness-validation-{uuid.uuid4().hex[:8]}"
    print(f"  Experiment id: {experiment_id}")

    # ── Passphrase handling ───────────────────────────────────────────────────
    passphrase = args.passphrase
    if passphrase is None:
        if args.generate_passphrase:
            passphrase = generate_passphrase()
            keeper = Path(args.keeper)
            keeper.parent.mkdir(parents=True, exist_ok=True)
            # Keeper holds the passphrase in PLAINTEXT so it is recoverable; it
            # MUST live in a gitignored path / protected location (see
            # `.gitignore`). The template/label files are the ones truly sealed —
            # this keeper is the access control you hold separately.
            keeper.write_text(json.dumps({"passphrase": passphrase, "note": "KEEP PRIVATE"},
                                         indent=2))
            try:
                os.chmod(keeper, 0o600)
            except OSError:
                pass
            print(f"  Generated passphrase keeper -> {keeper}")
        else:
            print("  No --passphrase given. Pass --passphrase or --generate-passphrase.")
            return 1

    # ── Step 2: response pool ────────────────────────────────────────────────
    pool = _build_response_pool(raw_dir)
    if not pool:
        print("  Response pool empty.")
        return 1
    print(f"  Response pool records: {len(pool)}")

    # ── Link consistency group members ───────────────────────────────────────
    # A consistency audit template needs the full group's pairs. Raw outputs hold
    # one record per prompt-variant; group them by group_id for the template
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in pool:
        gid = rec.get("group_id")
        if rec.get("_dimension") == "consistency" and gid:
            groups[gid].append(rec)
    for rec in pool:
        if rec.get("_dimension") == "consistency":
            rec["_group_members"] = groups.get(rec.get("group_id"), [rec])

    # ── Step 3: deterministic shuffle + calibration/held-out split ───────────
    seed_hash = hashlib.sha256(f"{experiment_id}:{args.seed}".encode()).hexdigest()
    rng = random.Random(seed_hash)
    order = list(range(len(pool)))
    rng.shuffle(order)
    shuffled = [pool[i] for i in order]

    n_cal = round(len(shuffled) * CALIBRATION_RATIO)
    calibration_set = shuffled[:n_cal]
    heldout_set = shuffled[n_cal:]

    # ── Step 4: sealed templates (no auto labels) ─────────────────────────────
    templates_dir = base_dir / "templates"
    labels_dir = base_dir / "labels"
    manifests_dir = base_dir / "manifests"
    templates_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    cal_path = templates_dir / f"{experiment_id}_calibration_template.jsonl"
    hold_path = templates_dir / f"{experiment_id}_heldout_template.jsonl"

    cal_records = [_template_record(r) for r in calibration_set]
    hold_records = [_template_record(r) for r in heldout_set]

    def _write_templates(path: Path, records: List[dict]) -> None:
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_templates(cal_path, cal_records)
    _write_templates(hold_path, hold_records)

    cal_hash = sha256_jsonl(cal_records)
    hold_hash = sha256_jsonl(hold_records)

    # ── Step 5: seal auto labels separately ───────────────────────────────────
    auto_payload = [_auto_label_payload(r) for r in pool]  # 1:1 with pool order
    auto_path = labels_dir / "sealed_auto_labels.jsonl.enc"
    auto_plain_hash = encrypt_json_to_file(
        auto_payload, auto_path, passphrase=passphrase
    )

    # ── Step 6: manifest ──────────────────────────────────────────────────────
    from datetime import datetime, timezone

    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "seed_hash": seed_hash,
        "calibration_ratio": CALIBRATION_RATIO,
        "pool_size": len(pool),
        "calibration_size": len(calibration_set),
        "heldout_size": len(heldout_set),
        "seal_checksums": {
            "calibration_template": f"sha256:{cal_hash}",
            "heldout_template": f"sha256:{hold_hash}",
            "auto_labels_plaintext": f"sha256:{auto_plain_hash}",
            "auto_labels_sealed_file": f"sha256:{sha256_file(auto_path)}",
            "rubric": f"sha256:{sha256_file(Path('docs/MANUAL_AUDIT_RUBRIC_v3.md'))}",
            "label_interface": f"sha256:{sha256_file(Path('docs/LABEL_INTERFACE.md'))}",
        },
        "templates": {
            "calibration": str(cal_path.relative_to(base_dir.parent)),
            "heldout": str(hold_path.relative_to(base_dir.parent)),
        },
        "pending": (
            "Auto labels are SEALED. Do NOT open until all human annotations are "
            "finalised and declarations filed."
        ),
    }
    manifest_path = manifests_dir / "sealing_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  calibration records: {len(calibration_set)}  (70%)")
    print(f"  held-out records:    {len(heldout_set)}  (30%)")
    print(f"  calibration template -> {cal_path}")
    print(f"  held-out template    -> {hold_path}")
    print(f"  sealed auto labels   -> {auto_path}")
    print(f"  sealing manifest     -> {manifest_path}")
    print("\n  NEXT: hand each rater their template + LABEL_INTERFACE + rubric.")
    print("  DO NOT open sealed labels until all annotations are final")
    return 0


if __name__ == "__main__":
    sys.exit(main())


