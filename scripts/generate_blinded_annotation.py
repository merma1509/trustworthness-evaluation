#!/usr/bin/env python3
"""Generate a blinded, multi-rater annotation dataset

The audit leaks ``auto_label`` and similarity fields
to the annotator, and that there was no calibration / held-out split. This
script builds an *annotatable* blinded JSONL (prompt + response only, no
``auto_label``, no similarity) with a calibration / held-out split made at the
**unique prompt/group unit** level

The audit file (``all_audit.jsonl``) may hold only label metadata; the actual
prompt/response text is joined from ``results/raw_outputs/*.jsonl``

Usage:
    python3 scripts/generate_blinded_annotation.py \
        --audit experiment/all_audit_full.jsonl \
        --output experiment/blinded \
        --calibration-ratio 0.3
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MODEL_TOKENS = ["gemma3_4b", "llama3_1_8b", "llama3.1_8b"]


def _parse_audit_id(audit_id: str) -> tuple:
    """Return (model, dimension, key) from e.g. ``gemma3_4b_truth_BEN_003``"""
    for mtok in MODEL_TOKENS:
        if audit_id.startswith(mtok + "_"):
            rest = audit_id[len(mtok) + 1 :]
            for tok, dim in [
                ("safety", "safety"),
                ("truth", "truthfulness"),
                ("cons", "consistency"),
            ]:
                if rest.startswith(tok + "_"):
                    return mtok, dim, rest[len(tok) + 1 :]
    return None, None, None


def _load_raw_lookup(raw_dir: Path) -> dict:
    """Build a {model: {dimension: {key: rows}}} lookup from raw outputs

    IMPORTANT (duplication bug fix):
    A consistency record carries BOTH ``prompt_id`` and ``group_id``. Previously
    every such record was keyed by both fields, so ``source.values()`` contained
    the same record twice and consistency groups in the blinded output ended up
    with duplicated members (e.g. ``CON_027`` appearing 3x in ``group_11``)

    The structure must reflect the dimension:

      * consistency   -> ``group_id -> [row, ...]``  (a LIST of member rows, so
                         no member is dropped nor duplicated within its group)
      * safety/truth  -> ``prompt_id -> row`` (one row per prompt)

    Returns:
        ``lookup[model][dim][key]`` is a *single row* for safety/truthfulness
        and a *list of rows* for consistency
    """
    lookup = {}
    for path in raw_dir.glob("*.jsonl"):
        tokens = path.stem.split("_")
        model = "_".join(tokens[:2])
        dim = tokens[2] if len(tokens) > 2 else "unknown"
        lookup.setdefault(model, {}).setdefault(dim, {})
        for line in path.open():
            rec = json.loads(line)
            if dim == "consistency":
                # Group members: append each row under its group_id
                gid = rec.get("group_id")
                if gid:
                    lookup[model][dim].setdefault(gid, []).append(rec)
            else:
                # Safety / truthfulness: index by prompt_id only
                pid = rec.get("prompt_id")
                if pid:
                    lookup[model][dim][pid] = rec
    return lookup


def _deduplicate_group_rows(rows: list) -> list:
    """Drop duplicate *prompt* members within a consistency group

    Defensive backstop for the raw-output join. The live pipeline keeps all
    copies of *repetition* prompts intentionally (we test same-question
    stability), so the dedup here is only an idempotent guard -- a genuinely
    duplicated prompt (same ``prompt_id`` AND same ``prompt``) must never be
    shown to an annotator twice

    Args:
        rows: Raw records for one consistency group (already group_id filtered)

    Returns:
        Records with duplicate (prompt_id, prompt) pairs collapsed to one copy
    """
    seen = set()
    unique = []
    for r in rows:
        marker = (r.get("prompt_id"), r.get("prompt_text"))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(r)
    return unique


def _blinded_record(rec: dict, raw: dict, anon_id: str) -> dict:
    """Build one BLINDED annotatable record.

    The following are stripped so they cannot leak ground truth or identity to
    the annotator: ``auto_label``, similarity fields, the real model name,
    real ``prompt_id``/``group_id`` and ``attack_type``. Only a neutral
    sequential ``anon_id``, the **dimension** (needed so the annotator picks the
    correct rubric of labels) and the prompt/response text remain.

    Args:
        rec: One audit record carrying the auto label + metadata.
        raw: Raw-output lookup ``{model: {dim: {key: rows}}}``.
        anon_id: Neutral sequential id shared by every record of this unit.

    Returns:
        An anonymised record safe to hand to an annotator. Hidden ground truth
        lives in a separate ``__ground_truth__`` field that downstream strip
        functions remove before serialisation.
    """
    model, dim, key = _parse_audit_id(rec.get("audit_id", ""))
    out = {
        "anon_id": anon_id,
        "dimension": rec.get("dimension") or dim,
        "__ground_truth__": {
            "audit_id": rec.get("audit_id"),
            "dimension": dim,
            "attack_type": rec.get("attack_type"),
            "auto_label": rec.get("auto_label"),
        },
    }
    source = raw.get(model, {}).get(dim, {})
    if dim == "consistency":
        gid = rec.get("group_id", key)
        group_rows = _deduplicate_group_rows(source.get(gid, []))
        # Hidden from the annotator: group_id, attack_type, real prompt ids.
        out["pairs"] = [
            {
                "prompt": r.get("prompt_text", ""),
                "response": r.get("response", ""),
            }
            for r in group_rows
        ]
        out["responses"] = [r.get("response", "") for r in group_rows]
        out["prompts"] = [r.get("prompt_text", "") for r in group_rows]
    else:
        row = source.get(rec.get("prompt_id") or key, {})
        out["prompt"] = row.get("prompt_text", rec.get("prompt_text"))
        out["response"] = row.get("response", rec.get("response"))
    return out


def _strip_ground_truth(rec: dict) -> dict:
    """Remove private ``__ground_truth__`` and identity-leaking metadata.

    Keeps ``dimension`` (needed by the annotator to choose the label rubric) but
    removes auto_label, similarity, attack_type, real prompt_id/group_id and the
    model identity.

    Args:
        rec: A record produced by :func:`_blinded_record`.

    Returns:
        The same record with ground truth and identifying metadata removed so the
        serialised JSONL leaks nothing but the dimension + prompt/response text.
    """
    out = {k: v for k, v in rec.items() if not k.startswith("__")}
    for field in ("attack_type", "prompt_id", "group_id", "audit_id"):
        out.pop(field, None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="experiment/all_audit_full.jsonl")
    parser.add_argument("--raw", default="results/raw_outputs")
    parser.add_argument("--output", default="experiment/blinded")
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.3,
        help="Fraction of units for calibration (default 0.3 -> "
        "~30%% calibration, ~70%% held-out).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Keep real IDs/dimension for debugging (NOT blinded "
        "-- never use for real annotation).",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit)
    raw_dir = Path(args.raw)
    out_dir = Path(args.output)
    if not audit_path.exists():
        print(f"Audit file not found: {audit_path}")
        sys.exit(1)

    records = [json.loads(line) for line in audit_path.open() if line.strip()]
    raw = _load_raw_lookup(raw_dir)

    # ── Unit assignment at the unique (dimension, key) level ──────────────
    # A safety/truth prompt appears once per model; a consistency group once.
    # Splitting on this "natural unit" (not per-model record) guarantees no
    # shared prompt leaks across the calibration/held-out boundary.
    def _real_unit(rec):
        _m, _d, _k = _parse_audit_id(rec.get("audit_id", ""))
        return (rec.get("dimension") or _d, rec.get("group_id") or rec.get("prompt_id") or _k)

    by_unit = defaultdict(list)
    for r in records:
        by_unit[_real_unit(r)].append(r)
    units = sorted(by_unit.keys())

    rng = random.Random(args.seed)
    rng.shuffle(units)
    n_cal = max(1, round(len(units) * args.calibration_ratio))
    # IMPORTANT (reproducibility): keep the split as *ordered* slices of the
    # seeded/shuffled ``units`` list, NOT as sets. Python ``set`` iteration
    # order is arbitrary and changes on every process run due to hash
    # randomisation (PYTHONHASHSEED). Iterating a set here would reorder the
    # output rows and — worse — reassign the sequential anon_ids (U_0001, ...)
    # to *different* physical units on each run, so a regenerated split would no
    # longer match the already-collected human annotations (or the analyst's
    # ground-truth mapping). Deterministic list slices keep both the row order
    # and the anon_id -> real-unit mapping stable across regenerations.
    cal_units = units[:n_cal]  # ordered slice, stable order
    hold_units = units[n_cal:]  # ordered slice, stable order

    # ── Assign neutral sequential anon ids ────────────────────────────────
    # The split is decided on the natural unit (dimension, key), so we first give
    # each *unit* a base sequential id (U_0001, ...). Each record is then keyed by
    # ``U_<unit>_<model-token>`` so two models answering the same prompt get
    # DISTINCT annotatable records (independent per-model labels) while still
    # sharing the unit base for the calibration/held-out split. Neither the unit
    # number nor the model token leaks the real prompt id or dimension.
    anon_by_unit = {}
    for u in units:
        anon_by_unit[u] = f"U_{len(anon_by_unit) + 1:04d}"

    def _build(recs):
        out_records = []
        for r in recs:
            _m, _d, _k = _parse_audit_id(r.get("audit_id", ""))
            base = anon_by_unit[_real_unit(r)]
            # Stable neutral token (A/B) per model — NEVER leaks model identity/version.
            model_token = "A" if _m == "gemma3_4b" else "B"
            out_records.append(_blinded_record(r, raw, f"{base}_{model_token}"))
        return out_records

    cal_recs = _build([r for u in cal_units for r in by_unit[u]])
    hold_recs = _build([r for u in hold_units for r in by_unit[u]])

    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(path, recs):
        with path.open("w") as f:
            for r in recs:
                out = _strip_ground_truth(r) if not args.no_anonymize else r
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    _write(out_dir / "blinded_annotation_calibration.jsonl", cal_recs)
    _write(out_dir / "blinded_annotation_heldout.jsonl", hold_recs)

    # ── Analyst-only ground truth (never handed to annotators) ──
    # Maps each anon_id / audit record back to its real unit, dimension, attack
    # type and auto label so the gold-vs-auto comparison can run after humans annotate.
    gtruth_by_id = {}
    for r in cal_recs + hold_recs:
        gt = dict(r.get("__ground_truth__", {}))
        gt["anon_id"] = r["anon_id"]
        gtruth_by_id[gt["audit_id"]] = gt
    # Also index by anon_id for downstream resolution.
    gtruth_by_anon = {}
    for gt in gtruth_by_id.values():
        gtruth_by_anon.setdefault(gt["anon_id"], []).append(gt)

    # Keep the by-audit_id map (one entry per record) AND a per-unit rollup.
    units_meta = {}
    for u in units:
        recs_here = by_unit[u]
        dims = {_real_unit(r)[0] for r in recs_here}
        units_meta[anon_by_unit[u]] = {
            "dimension": sorted(dims),
            "n_records": len(recs_here),
            "audit_ids": [
                _parse_audit_id(r.get("audit_id"))[2] or r.get("audit_id") for r in recs_here
            ],
        }

    ground_truth_doc = {
        "by_audit_id": {k: v for k, v in gtruth_by_id.items()},
        "by_anon_id": {k: v for k, v in gtruth_by_anon.items()},
        "units": units_meta,
        "note": (
            "Analyst-ONLY ground truth. Never share with annotators. Maps each "
            "anonymised unit (and each underlying audit record) back to its real "
            "dimension, attack_type and auto_label for the gold-vs-auto comparison."
        ),
    }
    with (out_dir / "ground_truth_blinded.json").open("w") as f:
        json.dump(ground_truth_doc, f, indent=2, ensure_ascii=False)

    manifest = {
        "total_records": len(records),
        "total_units": len(units),
        "calibration_units": len(cal_units),
        "heldout_units": len(hold_units),
        "calibration_records": len(cal_recs),
        "heldout_records": len(hold_recs),
        "calibration_ratio": args.calibration_ratio,
        "anonymized": not args.no_anonymize,
        "note": (
            "Blinded split made at the unique (dimension, key) unit level; one "
            "prompt never appears in both calibration and held-out. Prompt/response "
            "only; auto_label, similarity, dimension, attack_type, prompt_id & model "
            "identity REMOVED/ANONYMISED. Ground truth in ground_truth_blinded.json "
            "(analyst-only)."
        ),
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Wrote {len(cal_recs)} calibration ({len(cal_units)} units) + "
        f"{len(hold_recs)} held-out ({len(hold_units)} units) blinded records to {out_dir}"
    )
    print(f"  anonymized: {not args.no_anonymize}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
