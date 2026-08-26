#!/usr/bin/env python3
"""Measure human annotation time for a definitive auto/human cost ratio

The default ``human_time_per_label = 30s`` in ``compute_measurement_budget`` is
only a placeholder ("fill from a real timing study for the definitive ratio").
This script turns that into a **measured** value.

Design
------
A human annotator labels a sample of records from one of the annotation files
(safety/truthfulness -> ``correct|incorrect``, consistency -> ``consistent|inconsistent``)
and the wall-clock time of each label decision is recorded.  The median of the
per-label times is then written to ``results/human_timing_measurement.json``,
which ``paradigm_report.py`` auto-loads on subsequent runs so the cost ratio is
based on a real measurement instead of the 30s placeholder.

The annotator answers interactively one question at a time; the script measures
how long they take to read the prompt/response and commit to a label.

Usage
-----
    # Safety / truthfulness records (correct|incorrect):
    python3 scripts/measure_human_annotation_time.py \
        --input results/audit/all_audit.jsonl \
        --dimension safety \
        --sample 10

    # Consistency records (consistent|inconsistent):
    python3 scripts/measure_human_annotation_time.py \
        --input results/audit/all_audit.jsonl \
        --dimension consistency \
        --sample 10

    # Or take the median from all *.jsonl under a directory:
    python3 scripts/measure_human_annotation_time.py --all

Output
------
    results/human_timing_measurement.json :
        {
          "measured_at": "...",
          "method": "interactive_annotator_timing",
          "dimension": "safety",
          "measurement_validity": "MEASURED",
          "n_measured": 12,
          "median_seconds_per_label": 14.2,
          "mean_seconds_per_label": 15.1,
          "std_seconds_per_label": 4.0,
          "per_label_seconds": [ ... ],
          "notes": "..."
        }

    ``measurement_validity`` is set to ``"MEASURED"`` because this file is only
    ever produced by a real interactive timing study (a human at the keyboard
    labelling records one-by-one). ``paradigm_report.py`` checks this field to
    decide whether the value counts as genuinely measured or as a placeholder.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_jsonl

HUMAN_TIMING_PATH = Path("results/human_timing_measurement.json")
RAW_OUTPUTS_DIR = Path("results/raw_outputs")


def _full_multiline(text: str, indent: int = 0) -> str:
    """Return the FULL text, newlines re-indented so a long response stays readable.

    This intentionally does NOT truncate: annotators need to read the whole
    response to make a correct consistency/timing decision.
    """
    text = text or ""
    pad = " " * indent
    return text.replace("\n", "\n" + pad)


_RAW_INDEX = {}
def _load_raw_index() -> dict:
    """Lazily build {(dim, prompt_id): {prompt, response}} from raw outputs.

    Audit records only carry prompt_id / dimension / attack_type. To let an
    annotator actually READ the content being labelled, join back to the raw
    outputs produced by the evaluation run.
    """
    if _RAW_INDEX:
        return _RAW_INDEX
    for f in sorted(RAW_OUTPUTS_DIR.glob("*.jsonl")):
        dim = next(
            (d for d in ("safety", "truthfulness", "consistency") if f"_{d}" in f.name),
            None,
        )
        if dim is None:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                pid = r.get("prompt_id") or r.get("id")
                if not pid:
                    continue
                _RAW_INDEX[(dim, pid)] = {
                    "prompt": r.get("prompt_text") or r.get("prompt", ""),
                    "response": r.get("response", ""),
                    "expected": r.get("expected_behavior", ""),
                }
    return _RAW_INDEX


def _show_content(dimension: str, rec: dict) -> str:
    """Return a display string for a record using its RAW content if found."""
    idx = _load_raw_index()
    content = idx.get((dimension, rec.get("prompt_id") or rec.get("id")))
    if content:
        return (
            f"PROMPT:   {_full_multiline(content['prompt'])}\n"
            f"EXPECT:   {content['expected']}\n"
            f"RESPONSE: {_full_multiline(content['response'])}"
        )
    # Fallback: whatever fields the audit record carries.
    return (
        f"PROMPT:   {_full_multiline(rec.get('prompt_text') or rec.get('prompt'))}\n"
        f"RESPONSE: {_full_multiline(rec.get('response'))}"
    )


def run_timing_study(records, dimension: str, sample: int, seed: int = 42) -> dict:
    """Interactively time a human annotator on a sample of records."""
    rng = random.Random(seed)
    pool = [r for r in records if r.get("dimension") == dimension]
    if not pool:
        print(f"  No {dimension} records found in input.")
        return {}

    if sample and sample < len(pool):
        pool = rng.sample(pool, sample)

    if dimension == "consistency":
        label_help = "consistent / inconsistent"
    else:
        label_help = "correct / incorrect"

    times = []
    print(f"\n  Timing study for dimension '{dimension}' — {len(pool)} records.")
    print(f"  For each record, read the PROMPT + RESPONSE and type one of: {label_help}")
    print("  (You can also enter 's' to skip a record.)\n")

    for i, rec in enumerate(pool, 1):
        print(f"  --- Record {i}/{len(pool)} ---")
        if dimension != "consistency":
            print(f"  {_show_content(dimension, rec)}")
        else:
            # Consistency: show the group's paired responses (FULL text).
            for k in ("responses", "pairs"):
                val = rec.get(k)
                if val:
                    if isinstance(val, list):
                        for j, entry in enumerate(val):
                            if isinstance(entry, dict):
                                print(
                                    f"  [{j}] {_full_multiline(entry.get('prompt'), 6)}\n"
                                    f"       -> {_full_multiline(entry.get('response'), 12)}"
                                )
                            else:
                                print(f"  [{j}] {_full_multiline(entry, 6)}")
                    else:
                        print(f"  {k}: {_full_multiline(val, 6)}")
                    break

        start = time.monotonic()
        try:
            ans = input("  Label (enter): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted by user.")
            break
        elapsed = time.monotonic() - start

        if ans.lower() == "s":
            print("    (skipped)\n")
            continue
        times.append(elapsed)
        print(f"    -> {ans}  [{elapsed:.1f}s]\n")

    if not times:
        print("  No labels were timed — nothing to report.")
        return {}

    return {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "interactive_annotator_timing",
        "dimension": dimension,
        "measurement_validity": "MEASURED",
        "n_measured": len(times),
        "median_seconds_per_label": round(statistics.median(times), 2),
        "mean_seconds_per_label": round(statistics.mean(times), 2),
        "std_seconds_per_label": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
        "min_seconds_per_label": round(min(times), 2),
        "max_seconds_per_label": round(max(times), 2),
        "per_label_seconds": [round(t, 2) for t in times],
        "notes": (
            "Wall-clock median time per label across an interactive sample. "
            "Used by paradigm_report.py (Task 1.5) to replace the 30s placeholder "
            "with a MEASURED value for the definitive auto/human cost ratio."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/audit/all_audit.jsonl", help="Annotation JSONL source file"
    )
    parser.add_argument(
        "--dimension", default="safety", choices=["safety", "truthfulness", "consistency"]
    )
    parser.add_argument("--sample", type=int, default=8, help="Number of records to time (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(HUMAN_TIMING_PATH))
    args = parser.parse_args()

    records = load_jsonl(args.input)
    result = run_timing_study(records, args.dimension, args.sample, args.seed)
    if not result:
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved measured human timing to {out}")
    print(f"  Median: {result['median_seconds_per_label']:.1f}s/label  (n={result['n_measured']})")
    print(
        "  -> Run `scripts/paradigm_report.py --with-cost` to use this "
        "measured value in the cost ratio."
    )


if __name__ == "__main__":
    main()
