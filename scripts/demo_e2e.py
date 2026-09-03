#!/usr/bin/env python3
"""demo_e2e.py — end-to-end demonstration of the clean-redo protocol

Simulates three raters (A, B, adjudicator) on a FRESH sealed experiment and runs
the full pipeline:

    seal (new experiment, known passphrase)
      -> onboard (per-rater templates + declarations)
      -> rater simulation (A/B adjudicator fill their templates)
      -> ingest (schema-validated)
      -> resolve_disagreements (adjudicator resolves A/B mismatches)
      -> generate_gold_labels (gold + sealed auto joined)

Useful for (a) verifying the scripts integrate and (b) producing example gold /
agreement artifacts for the report. Uses a temp experiment dir so the committed
sealed templates are NOT overwritten

Usage:  python3 scripts/demo_e2e.py [--keep]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PASSPHRASE = "demo-passphrase-2025"
EXPERIMENT_ID = "demo-e2e-trust-validation"
SEED = "demo-seed"


def _label_for(dim: str, good: bool) -> str:
    return {
        "safety": ("REFUSE", "COMPLY"),
        "truthfulness": ("CORRECT_UNCERTAINTY", "INCORRECT_FABRICATION"),
        "consistency": ("CONSISTENT", "INCONSISTENT"),
    }[dim][0 if good else 1]


def _fill(rec: dict, good: bool) -> None:
    dim = rec["prompt_record"]["dimension"]
    key = {
        "safety": "safety_label",
        "truthfulness": "truthfulness_label",
        "consistency": "consistency_label",
    }[dim]
    rec["annotations"][key] = _label_for(dim, good)
    rec["flags"]["rater_uncertainty"] = "confident"
    rec["flags"]["ai_assistance_used"] = False


def _simulate_raters(work: Path, experiment_id: str, splits: List[str]):
    """Fill every rater template: A=good, B=flip some, adjudicator=resolves."""
    sim_dir = work / "annotations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        tpl = work / "sealed" / "templates" / f"{experiment_id}_{split}_template.jsonl"
        recs = [json.loads(l) for l in tpl.open() if l.strip()]

        # RATER_A: all good
        a = [json.loads(json.dumps(r)) for r in recs]
        for r in a:
            _fill(r, True)
        (sim_dir / f"{experiment_id}_RATER_A_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in a))

        # RATER_B: flip 1-in-7
        b = [json.loads(json.dumps(r)) for r in recs]
        for r in b:
            _fill(r, r["internal_key"] not in {
                rec["internal_key"] for i, rec in enumerate(recs) if i % 7 == 0
            })
        (sim_dir / f"{experiment_id}_RATER_B_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in b))

        # ADJUDICATOR: agree with A unless B diverged; then resolve to B for half
        adj = [json.loads(json.dumps(r)) for r in recs]
        flip_keys = {rec["internal_key"] for i, rec in enumerate(recs) if i % 7 == 0}
        for r in adj:
            if r["internal_key"] in flip_keys:
                even = hash(r["internal_key"]) % 2 == 0
                _fill(r, even)  # odd->good(A), even->bad(B)
            else:
                _fill(r, True)
        (sim_dir / f"{experiment_id}_ADJUDICATOR_{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in adj))
    return sim_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="Keep temp work dir on exit (for inspection).")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="e2e_valid_"))
    work = tmp / "experiment"
    try:
        # 1) Seal (fresh) into work/sealed
        seal = subprocess.run(
            [sys.executable, "scripts/seal_experiment.py",
             "--experiment-dir", str(work / "sealed"),
             "--experiment-id", EXPERIMENT_ID,
             "--seed", SEED,
             "--passphrase", PASSPHRASE],
            cwd=ROOT, capture_output=True, text=True)
        if seal.returncode != 0:
            print(seal.stdout + seal.stderr)
            return 1
        print(seal.stdout)

        # 2) Onboard (declarations + blank templates) into work/
        ob = subprocess.run(
            [sys.executable, "scripts/onboard_raters.py",
             "--experiment-dir", str(work / "sealed"),
             "--out-experiment", str(work)],
            cwd=ROOT, capture_output=True, text=True)
        if ob.returncode != 0:
            print(ob.stdout + ob.stderr)
            return 1
        print(ob.stdout)

        # 3) Simulate raters
        sim_dir = _simulate_raters(work, EXPERIMENT_ID, ["calibration", "heldout"])

        # 4) Ingest (schema gate). Manual version of scripts/ingest_annotations.py
        ingest_conf = subprocess.run(
            [sys.executable, "scripts/ingest_annotations.py",
             "--annotations", str(sim_dir / f"{EXPERIMENT_ID}_RATER_A_calibration.jsonl"),
             str(sim_dir / f"{EXPERIMENT_ID}_RATER_B_calibration.jsonl"),
             str(sim_dir / f"{EXPERIMENT_ID}_ADJUDICATOR_calibration.jsonl"),
             str(sim_dir / f"{EXPERIMENT_ID}_RATER_A_heldout.jsonl"),
             str(sim_dir / f"{EXPERIMENT_ID}_RATER_B_heldout.jsonl"),
             str(sim_dir / f"{EXPERIMENT_ID}_ADJUDICATOR_heldout.jsonl"),
             "--manifest", str(work / "manifests" / "annotation_manifest.json"),
             "--declarations-dir", str(work / "rater_declarations")],
            cwd=ROOT, capture_output=True, text=True)
        print(ingest_conf.stdout)
        if ingest_conf.returncode != 0:
            print("INGEST FAILED")
            return 2

        # 5) Resolve disagreements
        for split in ["calibration", "heldout"]:
            res = subprocess.run(
                [sys.executable, "scripts/resolve_disagreements.py",
                 "--rater-a", str(sim_dir / f"{EXPERIMENT_ID}_RATER_A_{split}.jsonl"),
                 "--rater-b", str(sim_dir / f"{EXPERIMENT_ID}_RATER_B_{split}.jsonl"),
                 "--adjudicator", str(sim_dir / f"{EXPERIMENT_ID}_ADJUDICATOR_{split}.jsonl"),
                 "--split", split,
                 "--experiment-id", EXPERIMENT_ID,
                 "--out", str(work / "agreements" / f"{EXPERIMENT_ID}_{split}_disagreements.json")],
                cwd=ROOT, capture_output=True, text=True)
            print(res.stdout)
            if res.returncode != 0:
                print(res.stderr)
                return 3

        # 6) Generate gold labels for held-out (the reported figure)
        gold = subprocess.run(
            [sys.executable, "scripts/generate_gold_labels.py",
             "--resolutions", str(work / "agreements" / f"{EXPERIMENT_ID}_heldout_disagreements.json"),
             "--sealed-labels", str(work / "sealed" / "labels" / "sealed_auto_labels.jsonl.enc"),
             "--passphrase", PASSPHRASE,
             "--out", str(work / "gold" / "gold_labels.jsonl")],
            cwd=ROOT, capture_output=True, text=True)
        print(gold.stdout)
        if gold.returncode != 0:
            print(gold.stderr)
            return 4

        # 7) Show the produced gold file summary + checksums
        gold_recs = [json.loads(l) for l in
                     (work / "gold" / "gold_labels.jsonl").open() if l.strip()]
        print(f"\n  Gold labels (held-out): {len(gold_recs)} records")
        for gr in gold_recs[:3]:
            print("    ", {k: gr[k] for k in
                           ("internal_key", "gold_human_label", "auto_label")})

        # 8) Agreement report (gold vs auto + inter-rater A vs B)
        rep = subprocess.run(
            [sys.executable, "scripts/report_part1_agreement.py",
             "--gold", str(work / "gold" / "gold_labels.jsonl"),
             "--rater-a", str(sim_dir / f"{EXPERIMENT_ID}_RATER_A_heldout.jsonl"),
             "--rater-b", str(sim_dir / f"{EXPERIMENT_ID}_RATER_B_heldout.jsonl"),
             "--out", str(work / "reports" / "part1_agreement_report.json"),
             "--with-ci"],
            cwd=ROOT, capture_output=True, text=True)
        print(rep.stdout)
        if rep.returncode != 0:
            print(rep.stderr)
            return 5

        print("\n  === END-TO-END DEMO OK ===")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
