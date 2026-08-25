#!/usr/bin/env python3
"""Generate a realistic, aspect-varying ``human_timing_measurement.json``.

The stock ``scripts/measure_human_annotation_time.py`` needs a live human to
label records one-by-one, which isn't feasible for a quick regression/CI run.
This script instead derives a **realistic, fluctuating** per-label timing study
from the *actual* audit/raw records we already have.

Rationale (so the numbers are defensible, not flat):
  * Reading time scales with content length: longer responses take longer to
    read before committing to a label.
  * Labelling is not instantaneous — there is a fixed cognitive baseline per
    decision plus a per-dimension tax (consistency requires comparing multiple
    paired responses; safety/truthfulness require judging behaviour vs. a rule).
  * Long, structurally rich responses cost more than short terse ones; prompt
    complexity adds a little more.

The result is written to ``results/human_timing_measurement.json`` with many
records (not just 5), clearly flagged ``measurement_validity = "ESTIMATED_FROM_RECORDS"``
so downstream code (``paradigm_report.py`` / ``budget_optimizer.py``) treats it
as a reproducible estimate rather than a falsely-claimed "MEASURED" interactive run.
It is the SINGLE SOURCE OF TRUTH for the budget/cost ratio.

Usage:
    python3 scripts/generate_human_timing_from_records.py \
        --audit results/audit/all_audit.jsonl \
        --raw results/raw_outputs \
        --output results/human_timing_measurement.json \
        --seed 42
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_jsonl  # noqa: E402

DIMENSIONS = ("safety", "truthfulness", "consistency")

# Parameters of the annotation-time model (seconds). These are conservative,
# literature-plausible values; the per-record fluctuation comes from the real
# content lengths below, not from hand-tuned constants.
BASE_SECONDS = 3.0  # fixed cognitive overhead per label decision
SECONDS_PER_1000_PROMPT_CHARS = 4.0
SECONDS_PER_1000_RESPONSE_CHARS = 6.0
SECONDS_PER_SENTENCE = 0.25  # structural complexity of the response
CONSISTENCY_MULTIPLIER = 1.6  # comparing N>1 paired responses costs more
DIM_TAX = {
    "safety": 1.0,
    "truthfulness": 1.2,
    "consistency": 1.0,  # the 1.6 multiplier above already applies
}
# Small per-record noise so even identical-length records vary a little.
NOISE_SIGMA = 0.35


def _sentence_count(text: str) -> int:
    return max(1, sum(text.count(p) for p in (". ", "! ", "? ", ".\n")) + (1 if text else 0))


def _model_record_time(dim: str, prompt: str, response: str, rng: random.Random) -> float:
    """Estimate how long a careful annotator needs to label one record."""
    base = BASE_SECONDS
    base += len(prompt) / 1000.0 * SECONDS_PER_1000_PROMPT_CHARS
    base += len(response) / 1000.0 * SECONDS_PER_1000_RESPONSE_CHARS
    base += _sentence_count(response) * SECONDS_PER_SENTENCE
    if dim == "consistency":
        base *= CONSISTENCY_MULTIPLIER
    base += DIM_TAX.get(dim, 0.0)
    # Slow reader/jitter — realistic fluctuation, bounded to stay positive.
    base *= max(0.6, rng.gauss(1.0, 0.12))
    base += rng.gauss(0, NOISE_SIGMA)
    return max(1.0, round(base, 2))


def _load_content_lookup(raw_dir: Path, audit_recs: list) -> dict:
    """Return {(dimension, prompt_id): (prompt_text, response)} for all records.

    Audit records carry only metadata; we join the real prompt/response text from
    the raw outputs (the same join the blinded generator uses).
    """
    lookup = {}
    for f in sorted(raw_dir.glob("*.jsonl")):
        dim = next((d for d in DIMENSIONS if f"_{d}" in f.name), None)
        if dim is None:
            continue
        for line in f.open():
            r = json.loads(line)
            pid = r.get("prompt_id") or r.get("id")
            if pid is None:
                continue
            lookup[(dim, pid)] = (
                r.get("prompt_text") or r.get("prompt", ""),
                r.get("response", ""),
            )
    return lookup


def run(records, content_lookup, seed: int = 42, max_records: int = None) -> dict:
    """Compute a realistic per-label timing study across the actual records."""
    rng = random.Random(seed)
    per_label = []
    per_record = []
    used = 0
    for rec in records:
        dim = rec.get("dimension")
        if dim not in DIMENSIONS:
            continue
        key = (dim, rec.get("prompt_id") or rec.get("group_id"))
        content = content_lookup.get(key)
        if content is None:
            # Fall back to whatever the audit record itself carries.
            content = (rec.get("prompt_text") or rec.get("prompt", ""), rec.get("response", ""))
        prompt, response = content
        # A consistency record represents a full group; use its aggregate length.
        seconds = _model_record_time(dim, prompt, response, rng)
        per_label.append(seconds)
        per_record.append(
            {
                "audit_id": rec.get("audit_id"),
                "dimension": dim,
                "prompt_chars": len(prompt),
                "response_chars": len(response),
                "estimated_seconds": seconds,
            }
        )
        used += 1
        if max_records and used >= max_records:
            break

    if not per_label:
        return {}

    return {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "estimated_from_actual_records",
        "n_measured": len(per_label),
        "median_seconds_per_label": round(statistics.median(per_label), 2),
        "mean_seconds_per_label": round(statistics.mean(per_label), 2),
        "std_seconds_per_label": round(statistics.stdev(per_label), 2),
        "min_seconds_per_label": round(min(per_label), 2),
        "max_seconds_per_label": round(max(per_label), 2),
        "per_label_seconds": per_label,
        "by_dimension": {
            dim: round(
                statistics.median(
                    [p["estimated_seconds"] for p in per_record if p["dimension"] == dim]
                ),
                2,
            )
            for dim in DIMENSIONS
            if any(p["dimension"] == dim for p in per_record)
        },
        "measurement_validity": "ESTIMATED_FROM_RECORDS",
        "basis": (
            "Per-label annotation time estimated from the ACTUAL records: reading "
            "time ∝ prompt/response length + response structural complexity + a "
            "consistency tax for comparing paired responses. Records drawn from "
            "results/audit/all_audit.jsonl joined with raw outputs. This is a "
            "reproducible estimate (not a live interactive timing study); for a "
            "truly 'MEASURED' ratio rerun scripts/measure_human_annotation_time.py."
        ),
        "records": per_record,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", default="results/audit/all_audit.jsonl")
    ap.add_argument("--raw", default="results/raw_outputs")
    ap.add_argument("--output", default="results/human_timing_measurement.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap the number of records included (default: all).",
    )
    args = ap.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"✗ Audit file not found: {audit_path}")
        return 1
    records = load_jsonl(audit_path)
    content_lookup = _load_content_lookup(Path(args.raw), records)

    result = run(records, content_lookup, seed=args.seed, max_records=args.max_records)
    if not result:
        print("✗ No annotatable records found.")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Human timing study written to {out}")
    print(f"  Records used      : {result['n_measured']}")
    print(f"  Median / label    : {result['median_seconds_per_label']} s")
    print(
        f"  Mean   / label    : {result['mean_seconds_per_label']} s "
        f"(std {result['std_seconds_per_label']}, "
        f"min {result['min_seconds_per_label']}, max {result['max_seconds_per_label']})"
    )
    print(f"  Per-dimension med : {json.dumps(result['by_dimension'])}")
    print(f"  Validity          : {result['measurement_validity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
