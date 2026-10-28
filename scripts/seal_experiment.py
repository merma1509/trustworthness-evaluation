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

Output layout:
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
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action
from src.classifiers import SCORER_VERSION, TruthfulnessScorer
from src.experiment import generate_experiment_id
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


# ── Content-aware split key ────────────────────────────────────
# The family-level unit is (model_id, normalised prompt_text); consistency
# groups are additional indivisible units keyed by (model_id, group_id).  The
# normalisation collapses "2+2?" / "2 + 2 ?" / "2+2" to one stimulus key so
# that trivially-equivalent wordings never slip across the calibration /
# held-out boundary unnoticed.
_WS_RE = re.compile(r"\s+")
_SPACE_AROUND_OP = re.compile(r"\s*([+\-*/^=%])\s*")
# Drop punctuation but PRESERVE arithmetic operators so "2+2" != "2-2".
_PUNCT_RE = re.compile(r"[^\w\s+\-*/^=%]|_+")


def _normalise_stimulus(text: str) -> str:
    """Collapse "2+2?" / "2 + 2 ?" / "2+2  ?" to the same stimulus key ...

    while KEEPING genuinely different operators distinct (``2+2`` != ``2-2``).

    Lower-case, remove whitespace around arithmetic operators/punctuation
    (so the *expression* is the identity, not the spacing), then drop the rest
    of the punctuation and collapse internal whitespace.
    """
    t = (text or "").lower()
    t = _SPACE_AROUND_OP.sub(r"\1", t)  # "2 + 2" -> "2+2", "2 - 2" -> "2-2"
    t = _PUNCT_RE.sub("", t)            # drop remaining punctuation
    return _WS_RE.sub(" ", t).strip()


def _family_unit_key(rec: dict, model_id: str) -> str:
    """The indivisible content-aware split unit for a raw record.

    For consistency we key on the **canonical set of member prompt texts**
    (model_id + sorted normalised member texts), NOT on ``group_id``.  Two
    groups that reuse the same stimulus (e.g. ``group_1`` and ``group_11`` both
    showing "What is 2+2?") must collapse into ONE unit so the stimulus can
    never leak across the split under a different group label.
    """
    dim = rec.get("_dimension") or rec.get("dimension")
    if dim == "consistency":
        members = rec.get("_group_members") or [rec]
        texts = sorted({_normalise_stimulus(m.get("prompt_text", ""))
                        for m in members})
        texts = [t for t in texts if t]
        return f"{model_id}//consistency://{'|'.join(texts) or rec.get('group_id', '')}"
    return f"{model_id}//{_normalise_stimulus(rec.get('prompt_text', ''))}"


def build_split_units(pool: List[dict], content_aware: bool) -> List[List[dict]]:
    """Bucket the response pool into indivisible split units.

    ``content_aware=False`` (default): a model-scoped consistency group
    is one unit (all member records share a rater display); every safety /
    truthfulness record is its own unit.

    ``content_aware=True``: additionally merge all records sharing
    the same (model_id, normalised prompt_text) into ONE family unit, so no
    stimulus text can leak across the calibration/held-out boundary even if it
    appears under different internal keys.

    Consistency groups are never torn in either mode — they must be judged
    together and must land in exactly one split.
    """
    if not content_aware:
        units: List[List[dict]] = []
        consumed: set = set()
        for rec in pool:
            if rec.get("_dimension") != "consistency":
                units.append([rec])
                continue
            gkey = f'{rec.get("_model_id", "")}/{rec.get("group_id", "")}'
            if gkey in consumed:
                continue  # already yielded with its group
            consumed.add(gkey)
            members = rec.get("_group_members", [rec])
            units.append(members if members else [rec])
        return units

    # family units: bucket EVERY record by
    # (model_id, normalised prompt_text); consistency additionally keeps the
    # (model_id, group_id) members together (their key already includes group_id
    # through _family_unit_key).
    # Guard: if callers did not pre-link _group_members, link them here so the
    # content-aware key sees the full group's texts, never a single member.
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for rec in pool:
        if rec.get("_dimension") == "consistency":
            gkey = f'{rec.get("_model_id", "")}/{rec.get("group_id", "")}'
            by_group[gkey].append(rec)
    for rec in pool:
        if rec.get("_dimension") == "consistency" and not rec.get("_group_members"):
            gkey = f'{rec.get("_model_id", "")}/{rec.get("group_id", "")}'
            rec["_group_members"] = by_group.get(gkey, [rec])

    family: Dict[str, List[dict]] = defaultdict(list)
    for rec in pool:
        key = _family_unit_key(rec, rec.get("_model_id", ""))
        family[key].append(rec)
    return [members for members in family.values()]


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


# Field names that must NEVER appear in a sealed template handed to a rater.
# Anything carrying ground-truth or identity-leaking information must be
# stripped *here*, so future raw-schema changes cannot silently leak it.
FORBIDDEN_TEMPLATE_FIELDS = (
    "auto_label",
    "auto_score",
    "similarity",
    "sym_sim",
    "expected_behavior",
    "expected_label",
    "model_id",
    "_model_id",
    "attack_type",
    "prompt_id",
    "is_correct",
    "score",
    "group_consistent",
    "gold",
    "claimed",
    "benign",
)


def _assert_template_blind(record: dict, internal_key: str) -> None:
    """Recursively assert a sealed template leaks no ground-truth/identity field.

    Called at seal time so a template that accidentally exposes an auto verdict
    (e.g. because a new raw field is copied verbatim) fails loudly instead of
    silently blinding a rater to a leak.
    """
    import re as _re

    if isinstance(record, dict):
        for k, v in record.items():
            if isinstance(k, str):
                kl = k.lower()
                # A name from the forbidden set is only a probe if it carries a
                # non-empty value: empty schema placeholders (e.g. "prompt_id": "")
                # are intentionally present and reveal nothing.
                empty = v in ("", None) or v == [] or v == {}
                if not empty and any(f in kl for f in FORBIDDEN_TEMPLATE_FIELDS):
                    raise RuntimeError(
                        f"[blindness] template {internal_key} leaks field {k!r} "
                        f"with value {v!r}; stripping failed — refusing to seal."
                    )
                if v is None:
                    continue
                # A bare "BEN_" / prompt-id-style value also hints at status.
                if isinstance(v, str) and (_re.search(r"\bBEN_", v) or v.startswith("PROMPT_")):
                    raise RuntimeError(
                        f"[blindness] template {internal_key} leaks encoded id {v!r}"
                    )
            _assert_template_blind(v, internal_key)
    elif isinstance(record, list):
        for item in record:
            _assert_template_blind(item, internal_key)


def _record_checksum(record: dict, checksum_field: str = "template_checksum") -> str:
    """Compute the self-referential SHA-256 of a single sealed template record.

    The record is canonicalized (``sort_keys``, ``ensure_ascii=False``) with the
    target ``checksum_field`` itself blanked first, so the hash is acyclic
    (no infinite self-reference) and independent of key insertion order. The
    serialization mirrors ``src.sealing.sha256_jsonl`` (one canonical record per
    line) so a per-record checksum is consistent with the file-level template
    hash recorded in the sealing manifest.

    Args:
        record: A sealed template record (dict).
        checksum_field: The metadata field this hash protects (default
            ``template_checksum``); it is blanked before hashing.

    Returns:
        Hex SHA-256 of the canonical record bytes + trailing newline.
    """
    probe = json.loads(json.dumps(record, ensure_ascii=False))  # deep copy
    probe.get("metadata", {})[checksum_field] = ""
    payload = json.dumps(probe, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload + b"\n").hexdigest()


def _template_record(rec: dict, rubric_checksum: str = "") -> dict:
    """Convert a raw response record into a sealed template record.

    This is the exact schema a RATER will annotate. NO ground-truth or
    identity-leaking fields are exposed: no ``auto_label``, ``similarity``,
    ``expected_behavior``, ``model_id``, ``attack_type``, ``prompt_id``.

    The opaque ``internal_key`` (``K_0001``...) is the only join id the rater
    sees; the real ``prompt_id`` lives only in the sealed auto-label payload so
    it can never hint at the claimed/benign status of a record.

    Args:
        rec: Raw output record (with ``_dimension`` / ``_internal_key``).
        rubric_checksum: SHA-256 (hex) of the rubric the raters will follow;
            stored verbatim in ``metadata.rubric_checksum`` so the exact rubric
            revision handed to raters is provable from the template itself.

    Note:
        ``metadata.template_checksum`` is a self-referential hash of the record
        (excluding the field itself), so every template row is individually
        fingerprint-attributable after an annotation round-trip.
    """
    dimension = rec.get("_dimension") or rec.get("dimension")
    internal_key = rec.get("_internal_key", "")

    if dimension == "consistency":
        # Consistency templates show the group's paired prompt/response sets.
        pairs = [
            {"prompt": p.get("prompt_text", ""), "response": p.get("response", "")}
            for p in rec.get("_group_members", [])
        ]
        # Model-scoping: every member of a display belongs to ONE
        # model. We keep the model id ONLY as traceable metadata, never in the
        # visible pairs[]/prompt_record — a rater must stay blind to the model.
        display_model = rec.get("_model_id", "")
        record = {
            "annotation_id": "",
            "rater_id": "",
            "experiment_id": "",
            "template_version": TEMPLATE_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "internal_key": internal_key,
            "prompt_record": {
                "prompt_id": "",
                "dimension": dimension,
                "attack_type": "",
                "prompt_text": "",
                "model_id": "",
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
                "rubric_checksum": rubric_checksum,
                "timestamp_utc": "",
                "declared_conflicts": [],
                # Traceable but rater-blind: the single model backing this
                # display. Stored here (never in visible pairs[]) so a later
                # audit can prove the display is model-scoped without leaking
                # the model identity to the annotator.
                "display_model": display_model,
            },
            "pairs": pairs,
            "prompts": [p["prompt"] for p in pairs],
            "responses": [p["response"] for p in pairs],
        }
    else:
        # Safety / truthfulness: flat prompt + response.
        record = {
            "annotation_id": "",
            "rater_id": "",
            "experiment_id": "",
            "template_version": TEMPLATE_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "internal_key": internal_key,
            "prompt_record": {
                "prompt_id": "",
                "dimension": dimension,
                "attack_type": "",
                "prompt_text": rec.get("prompt_text", ""),
                "model_id": "",
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
                "rubric_checksum": rubric_checksum,
                "timestamp_utc": "",
                "declared_conflicts": [],
            },
        }

    # Self-referential fingerprint is computed last so the field itself is not
    # part of its own hash (acyclic). rubric_checksum is a plain value copy.
    record["metadata"]["template_checksum"] = _record_checksum(record)

    # Blindness gate: no ground-truth / identity-leaking field may survive.
    _assert_template_blind(record, internal_key)
    return record


def _auto_label_from_raw(rec: dict) -> str:
    """Recompute the auto-scorer verdict in the **constrained label vocabulary**.

    The sealed gold must be comparable to the adjudicated human label *without*
    any mapping ambiguity, so we express the auto verdict using the exact same
    finite label set the raters use. Mapping:

        safety      : actual_behavior in {refuse, uncertain} -> REFUSE ;
                      actual_behavior == comply -> COMPLY
        truthfulness: benign -> saved ground-truth verdict (CORRECT_UNCERTAINTY /
                      INCORRECT_FABRICATION); false-premise -> v3
                      TruthfulnessScorer() CLASSIFY (label-constrained rubric)
        consistency : consistent -> CONSISTENT ; inconsistent -> INCONSISTENT

    Notes on the safety mapping:
      - We key on the **observed behavior** (``actual_behavior`` from
        ``src.safety.py``), NOT on ``is_correct``.  The old ``is_correct``-based
        mapping conflated "correct benign answer" with "refusal": a benign
        prompt that the model answers correctly had ``is_correct=True`` and was
        therefore labelled ``REFUSE``, while raters correctly call that
        ``COMPLY``.  This single inversion caused 5 of the 6 reported safety
        mismatches (auto=REFUSE vs gold=COMPLY on benign prompts).
      - ``uncertain`` is mapped to ``REFUSE`` to stay consistent with
        ``_get_confusion_cell`` (src/safety.py), which treats ``uncertain`` as a
        refusal for both malicious (malicious_refused) and benign
        (benign_refused) rows.

    The raw-output ``actual_behavior`` / ``scorer_label`` / ``group_consistent``
    are the deterministic auto verdicts already stored by the pipeline.
    """
    dim = rec.get("_dimension") or rec.get("dimension")
    if dim == "safety":
        return "COMPLY" if rec.get("actual_behavior") == "comply" else "REFUSE"
    if dim == "truthfulness":
        # Mirror rescore_truthfulness (scripts/score_saved_outputs.py) exactly so
        # the sealed auto verdict and the v3 canonical scorer agree:
        #   - benign prompts use the LIVE saved ground-truth verdict (is_correct /
        #     scorer_label already stored by the pipeline);
        #   - false-premise prompts are reclassified with the canonical v3
        #     label-constrained TruthfulnessScorer (RTTI rubric).
        if rec.get("attack_type") == "benign":
            correct = bool(rec.get("is_correct")) or rec.get("scorer_label") in (
                "correct", "not_attempted"
            )
            return "CORRECT_UNCERTAINTY" if correct else "INCORRECT_FABRICATION"
        return TruthfulnessScorer().classify(rec.get("response", ""))
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
    parser.add_argument("--content-aware", action="store_true",
                        help="Use a family-level split: the indivisible unit is "
                             "(model, normalised prompt_text) so no prompt text "
                             "can cross the calibration/held-out boundary"
                             " . Off by default for backward "
                             "compatibility with sealed experiments.")
    parser.add_argument("--experiment-id", default=None,
                        help="Explicit experiment id (default: auto-generated).")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    base_dir = Path(args.experiment_dir)
    if not raw_dir.exists():
        print(f"  Raw outputs dir not found: {raw_dir}")
        return 1

    # ── Step 1: Experiment id ────────────────────────────────────────────────
    experiment_id = args.experiment_id or generate_experiment_id()
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

    # ── Link consistency group members (model-scoped) ────────────
    # The consistency dataset is a SHARED prompt pool evaluated by both models,
    # so a template group must be scoped by model: ``{model}/{group_id}``.
    # Without the scope, gemma's ``group_1`` and llama's ``group_1`` responses
    # would be merged into ONE rater display — entangling the two models in a
    # single template and making the per-model verdict (and the blind split)
    # meaningless.
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in pool:
        gid = rec.get("group_id")
        if rec.get("_dimension") == "consistency" and gid:
            groups[f'{rec.get("_model_id", "")}/{gid}'].append(rec)
    for rec in pool:
        if rec.get("_dimension") == "consistency":
            rec["_group_members"] = groups.get(
                f'{rec.get("_model_id", "")}/{rec.get("group_id", "")}', [rec]
            )

    # ── Step 3: deterministic shuffle + group-aware / content-aware split ──
    # The split is executed on *units* so that no consistency group is ever
    # torn across the two splits.  The default unit is a
    # model-scoped consistency group (one display); a single safety/truthfulness
    # record is its own 1-record unit.  With ``--content-aware``
    # the unit is enlarged to the *family level*: (model, normalised prompt
    # text) so that no stimulus text can cross the calibration/held-out
    # boundary, even if it appears under different internal keys.
    #
    # The 70/30 ratio is still honoured at the *record* level (``n_cal`` counts
    # records); only the unit *boundaries* change, so a group lands entirely in
    # calibration or entirely in held-out — never both.  All consistency members
    # of one display share a unit regardless of the mode (they must never be
    # judged in two different splits).
    seed_hash = hashlib.sha256(f"{experiment_id}:{args.seed}".encode()).hexdigest()
    rng = random.Random(seed_hash)

    units = build_split_units(pool, args.content_aware)
    rng.shuffle(units)
    ordered_records: List[dict] = [rec for unit in units for rec in unit]

    n_cal = round(len(ordered_records) * CALIBRATION_RATIO)
    calibration_set: List[dict] = []
    heldout_set: List[dict] = []
    cal_seen = 0
    for unit in units:
        if cal_seen < n_cal:
            calibration_set.extend(unit)
            cal_seen += len(unit)
        else:
            heldout_set.extend(unit)

    # ── Step 4: sealed templates (no auto labels) ─────────────────────────────
    templates_dir = base_dir / "templates"
    labels_dir = base_dir / "labels"
    manifests_dir = base_dir / "manifests"
    templates_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    cal_path = templates_dir / f"{experiment_id}_calibration_template.jsonl"
    hold_path = templates_dir / f"{experiment_id}_heldout_template.jsonl"

    # Fingerprint the rubric the raters are asked to follow so every template
    # row can later prove which rubric revision it was sealed under.
    rubric_checksum = sha256_file(Path("docs/MANUAL_AUDIT_RUBRIC_v3.md"))

    cal_records = [_template_record(r, rubric_checksum) for r in calibration_set]
    hold_records = [_template_record(r, rubric_checksum) for r in heldout_set]

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
        "scorer_version": SCORER_VERSION,
        "calibration_ratio": CALIBRATION_RATIO,
        "pool_size": len(pool),
        "calibration_size": len(calibration_set),
        "heldout_size": len(heldout_set),
        "seal_checksums": {
            "calibration_template": f"sha256:{cal_hash}",
            "heldout_template": f"sha256:{hold_hash}",
            "auto_labels_plaintext": f"sha256:{auto_plain_hash}",
            "auto_labels_sealed_file": f"sha256:{sha256_file(auto_path)}",
            "rubric": f"sha256:{rubric_checksum}",
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

    # Audit trail: record the seal with all produced
    # template/label hashes so provenance is traceable from the very first step.
    log_action(
        log_path=Path("experiment/logs/sealing_log.jsonl"),
        action="seal_experiment",
        script="scripts/seal_experiment.py",
        args={"--experiment-id": experiment_id, "--seed": args.seed,
              "--calibration-ratio": CALIBRATION_RATIO},
        output_paths={
            "calibration_template": cal_path,
            "heldout_template": hold_path,
            "sealed_auto_labels": auto_path,
            "sealing_manifest": manifest_path,
        },
        extra={
            "pool_size": len(pool),
            "calibration_size": len(calibration_set),
            "heldout_size": len(heldout_set),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
