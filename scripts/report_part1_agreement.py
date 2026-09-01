#!/usr/bin/env python3
"""report_part1_agreement.py — Part 1 final agreement figures

Computes the held-out agreement statistics that go into the paper:

  1. Gold (adjudicated human label) VS auto-scorer label:
        overall + per-dimension agreement rate, Cohen's Kappa (+bootstrap CI),
        weighted kappa, and per-label precision/recall/F1.
  2. Inter-rater agreement between RATER_A and RATER_B:
        agreement rate + Cohen's Kappa per dimension.

Both inputs use the SAME constrained label vocabulary (REFUSE/COMPLY, ...), so
gold-vs-auto comparison is an exact string match with no mapping ambiguity.

Usage:
    python3 scripts/report_part1_agreement.py \
        --gold experiment/gold/gold_labels.jsonl \
        --rater-a experiment/annotations/{exp}_RATER_A_heldout.jsonl \
        --rater-b experiment/annotations/{exp}_RATER_B_heldout.jsonl \
        --out experiment/reports/part1_agreement_report.json \
        --with-ci
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agreement import compute_agreement, kappa_bootstrap_ci


def _load_gold(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _per_dimension(records: List[dict], human_key: str, auto_key: str):
    """Compute agreement per dimension from gold records."""
    by_dim: Dict[str, List[tuple]] = defaultdict(list)
    for r in records:
        h = r.get(human_key)
        a = r.get(auto_key)
        if h is not None and a is not None:
            by_dim[r.get("dimension", "unknown")].append((h, a))
    out = {}
    for dim in sorted(by_dim):
        hs = [p[0] for p in by_dim[dim]]
        as_ = [p[1] for p in by_dim[dim]]
        out[dim] = compute_agreement(hs, as_)
        out[dim]["n"] = len(hs)
    return out


def gold_vs_auto(gold_records: List[dict], with_ci: bool) -> Dict:
    """Gold (human-adjudicated) vs auto label agreement."""
    overall = compute_agreement(
        [r["gold_human_label"] for r in gold_records],
        [r["auto_label"] for r in gold_records],
        with_ci=with_ci,
    )
    per_dim = _per_dimension(gold_records, "gold_human_label", "auto_label")

    result = {
        "n_records": len(gold_records),
        "overall": overall,
        "per_dimension": per_dim,
    }
    if with_ci:
        hs = [r for r in gold_records if r.get("gold_human_label") and r.get("auto_label")]
        result["overall"]["kappa_ci"] = kappa_bootstrap_ci(
            [r["gold_human_label"] for r in hs],
            [r["auto_label"] for r in hs],
        )
    return result


def inter_rater(a_path: Path, b_path: Path) -> Dict:
    """Inter-rater agreement between RATER_A and RATER_B."""
    def _load(path: Path) -> Dict[str, tuple]:
        out = {}
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            dim = rec.get("prompt_record", {}).get("dimension", "")
            ann = rec.get("annotations") or {}
            key = {
                "safety": "safety_label",
                "truthfulness": "truthfulness_label",
                "consistency": "consistency_label",
            }.get(dim)
            label = ann.get(key, "") if key else ""
            out[rec.get("internal_key", "")] = (dim, label or None)
        return out

    a_map = _load(a_path)
    b_map = _load(b_path)

    by_dim: Dict[str, List[tuple]] = defaultdict(list)
    for k, (a_dim, la) in a_map.items():
        if k not in b_map:
            continue
        b_dim, lb = b_map[k]
        if la is None or lb is None:
            continue  # uncertain / not labelled don't cast a vote
        dim = a_dim or b_dim or "unknown"
        by_dim[dim].append((la, lb))

    result = {}
    for dim in sorted(by_dim):
        hs = [p[0] for p in by_dim[dim]]
        as_ = [p[1] for p in by_dim[dim]]
        result[dim] = compute_agreement(hs, as_)
        result[dim]["n"] = len(hs)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--rater-a", required=True)
    parser.add_argument("--rater-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-ci", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"  ✗ Gold file not found: {gold_path}")
        return 1
    gold = _load_gold(gold_path)

    report = {
        "gold_vs_auto": gold_vs_auto(gold, args.with_ci),
        "inter_rater_A_vs_B": inter_rater(Path(args.rater_a), Path(args.rater_b)),
        "note": (
            "Labels use the constrained vocabulary (REFUSE/COMPLY, ...). "
            "gold_vs_auto compares the adjudicated human label to the sealed "
            "auto-scorer label; inter_rater compares RATER_A to RATER_B."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Human-readable summary ─────────────────────────────────────────────
    gva = report["gold_vs_auto"]
    print("  GOLD vs AUTO (held-out):")
    for dim in ["overall"] + sorted(gva["per_dimension"]):
        s = (gva["overall"] if dim == "overall"
             else gva["per_dimension"].get(dim, {}))
        if not s or "n" not in s:
            continue
        k = s.get("cohens_kappa", 0)
        ci = s.get("kappa_ci")
        ci_txt = (f" CI=[{ci['ci_lower']},{ci['ci_upper']}]" if ci else "")
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={k}{ci_txt}")

    ir = report["inter_rater_A_vs_B"]
    print("  INTER-RATER (A vs B):")
    for dim, s in ir.items():
        print(f"    {dim:<12} n={s.get('n'):<4} agreement={s.get('agreement_rate'):<6} "
              f"kappa={s.get('cohens_kappa')}")

    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
