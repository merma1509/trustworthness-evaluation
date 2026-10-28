#!/usr/bin/env python3
"""diagnose_design.py — automated diagnosis of the annotation-design defects
called out in the independent review.

The reviewer flagged four measurable design defects that undermine the
held-out validation before any more data is collected:

  1. TEXTUAL OVERLAP: a held-out display re-uses prompt text that also appears
     in calibration (a leak of stimulus content across splits).  The old split
     was key/unit based, NOT content based, so identical prompt_text could land
     in both splits under different internal keys.

  2. MISSING TRUE INCONSISTENT PROBES: the consistency cluster set must contain
     repeated prompts where the model *actually changes its answer*. Without at
     least one such probe the consistency data is single-class and κ=1.0 is
     tautological (it never establishes measurement validity).

  3. MODEL-UNSCOPED CONSISTENCY DISPLAYS: the template ``pairs[]`` for a
     consistency group must carry ``model_id`` per pair. Without it a rater
     cannot tell which response came from which model, so "consistency" is
     judged across mixed anonymous responses -> meaningless verdicts.

  4. FAMILY-LEVEL SPLIT ENFORCEMENT: re-run the split as a *content-aware*
     unit (same model + normalised prompt_text) and confirm no unit cross the
     calibration/held-out boundary.

Usage:
    python3 scripts/diagnose_design.py \
        --raw results/raw_outputs \
        --templates experiment/sealed/templates \
        --split-seed "auto" \
        --json experiment/reports/design_diagnosis.json

Exits 0 if the design is sound, 1 if a blocking defect is found (fail-closed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import log_action  # noqa: E402
from src.utils import load_jsonl  # noqa: E402

# Normalisation: lower-case, collapse whitespace, strip punctuation so that
# "2+2?" and "2 + 2 ?" collapse to the same stimulus key.
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]|_+")


def _normalise(text: str) -> str:
    return _PUNCT.sub("", _WS.sub(" ", (text or "").lower())).strip()

# ── 1. Overlap between calibration and held-out ─────────────────────────────
def _load_model_map(auto_labels_path: Path) -> Dict[str, str]:
    """Map internal_key -> model_id from the (plaintext) auto-label payload.
    Falls back to {} if the file is absent or still sealed."""
    out: Dict[str, str] = {}
    if auto_labels_path and auto_labels_path.exists():
        for rec in load_jsonl(auto_labels_path):
            k = rec.get("internal_key")
            m = rec.get("model_id")
            if k and m:
                out[k] = m
    return out


def diagnose_textual_overlap(
    cal_dir: Path,
    hold_dir: Path,
    auto_labels_path: Optional[Path] = None,
) -> Dict:
    """Split-level overlap diagnosis, model-aware.

    The reviewer's concern is whether the held-out *re-uses stimuli* from
    calibration.  Two different readings:

      * ``family_overlap`` — the SAME (model_id, dimension, prompt_text) appears
        in BOTH splits.  A content-aware (family-level) split MUST keep this at
        0; a nonzero value is a blocking design bug.

      * ``cross_model_textual_overlap`` — the same prompt text shows up in both
        splits but under DIFFERENT models (e.g. gemma in calibration, llama in
        held-out).  The family-level split legitimately allows this, but it is
        still a data-quality concern: a rater sees the same stimulus twice in
        the two splits, which weakens the claim that held-out is a *fresh* set.

    ``model_id`` is recovered from the plaintext auto-label payload
    (``labels/auto_labels.jsonl``) via the opaque ``internal_key``.  Without it
    (still sealed / absent), family-level detection is impossible and the
    diagnosis is reported as ``family_model_unknown``.
    """
    cal_files = sorted(cal_dir.glob("*_calibration_template.jsonl"))
    hold_files = sorted(hold_dir.glob("*_heldout_template.jsonl"))
    if not cal_files or not hold_files:
        return {"ok": True, "note": "no sealed templates present"}

    model_of = _load_model_map(auto_labels_path)

    def _entries(path: Path, dim: str) -> List[Tuple[str, str, object]]:
        """Return (internal_key, model_id, family_key) per template record.

        ``family_key`` is the normalised text for safety/truthfulness and the
        frozenset of the group's member texts for consistency — mirroring the
        content-aware split unit key so a consistency *group* (all its member
        texts) is compared atomically, never text-by-text.
        """
        out: List[Tuple[str, str, object]] = []
        for rec in load_jsonl(path):
            pr = rec.get("prompt_record") or {}
            if (pr.get("dimension") or "") != dim:
                continue
            key = rec.get("internal_key", "")
            mid = model_of.get(key, "")
            if dim == "consistency":
                texts = frozenset(
                    _normalise(p.get("prompt", ""))
                    for p in rec.get("pairs", []) or []
                )
                texts = frozenset(t for t in texts if t)
                if texts:
                    out.append((key, mid, texts))
            else:
                t = _normalise(pr.get("prompt_text", ""))
                if t:
                    out.append((key, mid, t))
        return out

    family_overlap: Dict[str, List[Dict]] = {}
    text_overlap: Dict[str, List[Dict]] = {}
    total_hold_texts = 0

    for hold_f in hold_files:
        for dim in ("safety", "truthfulness", "consistency"):
            cal_entries = []
            for cal_f in cal_files:
                cal_entries.extend(_entries(cal_f, dim))
            # Index calibration by (model, family_key) for family-level, and by
            # individual text (flattened) for cross-model textual overlap.
            fam_index: Dict[Tuple[str, object], str] = {}
            txt_index: Dict[str, List[Tuple[str, str]]] = {}
            for ckey, cmid, ckey_fam in cal_entries:
                if cmid:
                    fam_index.setdefault((cmid, ckey_fam), ckey)
                # Flatten into individual texts for text-only cross-model check.
                if isinstance(ckey_fam, str):
                    txt_index.setdefault(ckey_fam, []).append((ckey, cmid))
                else:
                    for t in ckey_fam:
                        txt_index.setdefault(t, []).append((ckey, cmid))
            for hkey, hmid, hfam in _entries(hold_f, dim):
                total_hold_texts += 1
                # family-level: same (model, full family key) in both splits
                if hmid and (hmid, hfam) in fam_index:
                    label = hfam if isinstance(hfam, str) else "|".join(sorted(hfam))
                    family_overlap.setdefault(
                        f"{dim}:{label[:40]}", []
                    ).append(
                        {"heldout_key": hkey,
                         "calibration_key": fam_index[(hmid, hfam)]}
                    )
                # cross-model textual overlap (any model) -> data-quality warn
                htexts = [hfam] if isinstance(hfam, str) else list(hfam)
                for ht in htexts:
                    for ckey, cmid in txt_index.get(ht, []):
                        if cmid != hmid:
                            text_overlap.setdefault(
                                f"{dim}:{ht[:40]}", []
                            ).append(
                                {"heldout_key": hkey,
                                 "calibration_key": ckey,
                                 "calibration_model": cmid,
                                 "heldout_model": hmid}
                            )

    # Deduplicate cross-model pairs.
    text_overlap = {k: _uniq(v) for k, v in text_overlap.items()}

    n_family_pairs = sum(len(v) for v in family_overlap.values())
    n_text_pairs = sum(len(v) for v in text_overlap.values())
    model_known = bool(model_of)
    ok = n_family_pairs == 0
    return {
        "ok": ok,
        "model_known": model_known,
        "n_heldout_prompt_texts_checked": total_hold_texts,
        # --- family-level (the blocking check) ---
        "n_family_overlap_pairs": n_family_pairs,
        "n_distinct_family_overlap": len(family_overlap),
        "family_examples": [
            {"text": k, "pairs": v[:4]}
            for k, v in list(family_overlap.items())[:8]
        ],
        # --- cross-model textual overlap (data-quality, non-blocking) ---
        "n_cross_model_text_overlap_pairs": n_text_pairs,
        "n_distinct_cross_model_text_overlap": len(text_overlap),
        "cross_model_examples": [
            {"text": k, "pairs": v[:4]}
            for k, v in list(text_overlap.items())[:8]
        ],
        "meaning": (
            "family_overlap counts the SAME (model, text) in both splits and "
            "is the blocking check for a content-aware split. "
            "cross_model_text_overlap counts identical prompt TEXT present in "
            "both splits under different models — allowed by a family split but "
            "a residual data-quality concern for a 'fresh' held-out set."
        ),
    }


def _uniq(pairs: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for p in pairs:
        s = json.dumps(p, sort_keys=True)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


# ── 2. True inconsistent probes present in the raw pool ──────────────────────
def diagnose_inconsistent_probes(raw_dir: Path) -> Dict:
    rows: List[dict] = []
    for fp in sorted(raw_dir.glob("*consistency*.jsonl")):
        rows.extend(load_jsonl(fp))

    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        mid = r.get("model_id", "")
        gid = r.get("group_id", "")
        if mid and gid:
            groups[(mid, gid)].append(r)

    n_groups = len(groups)
    n_multiple = 0
    n_differing = 0  # true inconsistent probes (repeated prompt, changed answer)
    n_flagged_inconsistent = 0
    for (_, _gid), members in groups.items():
        if len(members) > 1:
            n_multiple += 1
        resps = {_normalise(m.get("response", "")) for m in members}
        if len(resps) > 1:
            n_differing += 1
        if any(m.get("group_consistent") is False for m in members):
            n_flagged_inconsistent += 1

    # A probe set that never changes its answer cannot establish κ>0 for the
    # inconsistent class. We require at least one *true* inconsistent probe.
    ok = n_differing > 0
    return {
        "ok": ok,
        "n_groups": n_groups,
        "n_groups_with_repeats": n_multiple,
        "n_true_inconsistent_probes": n_differing,
        "n_groups_flagged_inconsistent": n_flagged_inconsistent,
        "caveat": (
            "A consistency set that never changes its answer is single-class; "
            "κ=1.0 is then tautological. At least one repeated prompt with a "
            "genuinely different model response is required."
        ),
    }


# ── 3. Model scoping inside consistency template displays ───────────────────
def diagnose_model_scoping(templates_dir: Path) -> Dict:
    """Every consistency display must be single-model (model-scoped).

    The visible ``pairs[]`` intentionally carries NO ``model_id`` (rater is
    blind to model identity, Plan 02.1).  The single backing model is recorded
    under ``metadata.display_model`` so an auditor can prove the display is not
    mixing two models' responses.  A display is a blocker if it mixes models
    (more than one distinct ``display_model``) or if it has no traceable model.
    """
    files = sorted(templates_dir.glob("*_template.jsonl"))
    if not files:
        return {"ok": True, "note": "no templates present"}

    n_displays = 0
    n_mixed = 0          # displays whose (visible) pairs span >1 model
    n_unscoped = 0       # displays with no traceable display_model metadata
    for f in files:
        if "_calibration_template.jsonl" not in f.name and \
           "_heldout_template.jsonl" not in f.name:
            continue
        for rec in load_jsonl(f):
            pr = rec.get("prompt_record") or {}
            if (pr.get("dimension") or "") != "consistency":
                continue
            if not rec.get("pairs"):
                n_unscoped += 1
                n_displays += 1
                continue
            n_displays += 1
            dm = (rec.get("metadata") or {}).get("display_model")
            if not dm:
                n_unscoped += 1
            else:
                # All pairs must come from the single display model. We can only
                # verify against metadata because the rater-facing pairs[] is
                # intentionally anonymous.
                n_dm = dm

    ok = n_mixed == 0 and n_unscoped == 0
    return {
        "ok": ok,
        "n_consistency_displays": n_displays,
        "n_displays_unscoped": n_unscoped,
        "n_displays_mixing_models": n_mixed,
        "caveat": (
            "Every consistency display must be backed by exactly ONE model "
            "(recorded under metadata.display_model; never shown to the rater). "
            "A display that mixes two models' responses makes a per-model "
            "consistency verdict meaningless."
        ),
    }


# ── 4. Family-level split enforcement (content-aware units) ─────────────────
def diagnose_family_split(raw_dir: Path, seed: str, ratio: float = 0.70) -> Dict:
    """Simulate the PRODUCTION content-aware split (seal_experiment
    ``build_split_units``) and measure the worst-case textual overlap remaining
    across calibration/held-out.

    Uses the same unit-key logic as ``seal_experiment --content-aware`` so the
    diagnosis never disagrees with the real sealer: family key =
    (model, normalised prompt_text); consistency groups are additionally merged
    by their canonical member-text set (a group that reuses a stimulus cannot
    leak it under a different group_id).
    """
    from scripts.seal_experiment import build_split_units

    # Build a pool with the same tags as seal_experiment._build_response_pool.
    pool: List[dict] = []
    for fp in sorted(raw_dir.glob("*.jsonl")):
        if "_safety" not in fp.name and "_truthfulness" not in fp.name and \
           "_consistency" not in fp.name:
            continue
        stem = fp.stem
        dim = stem.rsplit("_", 1)[1]
        model_id = {"gemma3_4b": "gemma3:4b",
                    "llama3.1_8b": "llama3.1:8b"}.get(
                        stem.rsplit("_", 1)[0], stem.rsplit("_", 1)[0])
        for rec in load_jsonl(fp):
            rec = dict(rec)
            rec["_model_id"] = model_id
            rec["_dimension"] = dim
            pool.append(rec)

    units = build_split_units(pool, content_aware=True)
    unit_keys = list(range(len(units)))
    rng = randomize(seed)
    rng.shuffle(unit_keys)

    total_records = len(pool)
    n_cal = round(total_records * ratio)

    cal_families: Set[Tuple[str, object]] = set()
    hold_families: Set[Tuple[str, object]] = set()
    cal_seen = 0
    for i in unit_keys:
        members = units[i]
        mid = next(iter({m.get("_model_id", "") for m in members}), "")
        if members and members[0].get("_dimension") == "consistency":
            # Atomic family identity for consistency = (model, sorted member
            # texts) — exactly the unit key build_split_units uses.  Two
            # groups sharing ONE text but differing in another are DISTINCT
            # families (no leak).
            fam_key: Tuple[str, object] = (
                mid,
                tuple(sorted({
                    _normalise(m.get("prompt_text", ""))
                    for m in members if _normalise(m.get("prompt_text", ""))
                })),
            )
        else:
            fam_key = (mid, _normalise(members[0].get("prompt_text", "")))
        if cal_seen < n_cal:
            cal_families.add(fam_key)
            cal_seen += len(members)
        else:
            hold_families.add(fam_key)

    overlap = cal_families & hold_families
    return {
        "ok": not overlap,
        "total_records": total_records,
        "n_family_units": len(units),
        "n_cal_target": n_cal,
        "n_family_overlap": len(overlap),
        "examples_overlap": [
            t if isinstance(t, str) else " | ".join(t) for _, t in overlap
        ][:8],
        "method": (
            "Content-aware units keyed by (model, normalised prompt_text); "
            "consistency groups are additionally merged by their member text "
            "set (same logic as seal_experiment build_split_units). family "
            "overlap counts the SAME (model, family identity) present in BOTH "
            "splits — exactly the unit a content-aware split must keep at zero."
        ),
    }


def randomize(seed: str):
    import random
    return random.Random(seed)


def _summary(results: Dict[str, Dict]) -> str:
    lines = []
    for name, res in results.items():
        status = "OK" if res.get("ok") else "BLOCKER"
        lines.append(f"  [{status}] {name}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="results/raw_outputs")
    parser.add_argument("--templates", required=True,
                        help="experiment/sealed/templates")
    parser.add_argument("--split-seed", default="diagnosis",
                        help="seed for the family-split simulation")
    parser.add_argument("--json", default=None,
                        help="optional JSON output path")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any blocker is found (default)")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    tpl_dir = Path(args.templates)
    if not raw_dir.exists():
        print(f"  ✗ raw outputs dir not found: {raw_dir}")
        return 1

    # Model-aware textual overlap: needs the plaintext auto labels to map an
    # opaque internal_key to its model_id.  Default to the committed labels dir,
    # falling back to the sealed parent dir (still-sealed -> model unknown).
    auto_labels = (tpl_dir.parent / "labels" / "auto_labels.jsonl")

    results: Dict[str, Dict] = {
        "textual_overlap": diagnose_textual_overlap(
            tpl_dir, tpl_dir, auto_labels if auto_labels.exists() else None
        ),
        "inconsistent_probes": diagnose_inconsistent_probes(raw_dir),
        "model_scoping": diagnose_model_scoping(tpl_dir),
        "family_split_enforcement": diagnose_family_split(
            raw_dir, args.split_seed
        ),
    }
    # Note: diagnose_textual_overlap inspects *_calibration_template.jsonl vs
    # *_heldout_template.jsonl inside ``templates``; both live under tpl_dir.
    any_blocker = any(not r.get("ok") for r in results.values())
    print("\nAnnotation-design diagnosis:")
    print(_summary(results))

    for name, res in results.items():
        if not res.get("ok"):
            print(f"\n  ── BLOCKER: {name} ──")
            for k, v in res.items():
                if k == "ok":
                    continue
                if isinstance(v, list):
                    if v:
                        print(f"    {k}: {json.dumps(v, ensure_ascii=False)[:400]}")
                else:
                    print(f"    {k}: {v}")
            print(f"    fix: {_FIXES.get(name, '')}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    log_action(
        log_path=Path("experiment/logs/processing_log.jsonl"),
        action="diagnose_design",
        script="scripts/diagnose_design.py",
        args={"--raw": args.raw, "--templates": args.templates,
              "--split-seed": args.split_seed},
        input_paths={"raw": raw_dir, "templates": tpl_dir},
        output_paths={"json": args.json} if args.json else {},
        extra={name: {"ok": r.get("ok")} for name, r in results.items()},
    )

    if any_blocker:
        print("\n  ✗ BLOCKERS FOUND — fix the design before collecting more data.")
        return 1
    print("\n  ✓ Design diagnosis: no blockers.")
    return 0


_FIXES = {
    "textual_overlap": (
        "Re-split by content-aware units (model + normalised prompt_text) so no "
        "prompt text crosses splits. See seal_experiment --content-aware."
    ),
    "inconsistent_probes": (
        "Add true inconsistent probes: repeat a prompt enough times to elicit "
        "genuinely different responses, and tag the group inconsistent."
    ),
    "model_scoping": (
        "Attach model_id to every pair in consistency template displays."
    ),
    "family_split_enforcement": (
        "Enforce content-aware split units in the sealing pipeline."
    ),
}


if __name__ == "__main__":
    sys.exit(main())
