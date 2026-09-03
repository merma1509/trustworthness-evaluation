#!/usr/bin/env python3
"""generate_expected_results.py — snapshot the current committed reports into
results/expected_results.json.

This is PLAN PART2 §2.5.2 step 11: the committed file of expected values that
`verify_results.py --expected` checks regenerated reports against.

Only run this when a NEW reference result set is intentionally published
(i.e. after `make experiment-reproduce` yields the numbers you want to lock).
It snapshots:

  * TrustScore + per-dimension scores (from scores_report.json / trustscore_report.json)
  * bootstrap CIs (from ci_report.json)
  * per-configuration ranking winners (from ranking_stability.json)

Usage:
    python3 scripts/generate_expected_results.py \
        --results results --output results/expected_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audit_log import log_action  # noqa: E402

DEFAULT_WEIGHTS = {"w_s": 0.4, "w_t": 0.35, "w_c": 0.25}


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_expected(
    scores: dict,
    ci: dict,
    ranking: dict,
    trust: dict,
) -> Dict:
    """Assemble the expected-results structure from the committed reports."""

    trust_by = trust.get("trustscore_by_model", {}) if trust else {}
    models: List[str] = (scores or {}).get("models", [])

    expected: Dict = {
        "pipeline": (
            "scripts/compute_scores.py + scripts/compute_ci.py + "
            "scripts/compute_ranking_stability.py + scripts/generate_results_json.py"
        ),
        "generated_from": (
            "Committed reference reports (results/*_report.json + "
            "ranking_stability.json). verify_results --expected checks "
            "regenerated reports against these."
        ),
        "models": models,
        "trustscore": {},
        "ci": {},
        "ranking": {"configurations": []},
    }

    per_model = (scores or {}).get("per_model", {})
    for m in models:
        pm = per_model.get(m, {})
        dims = {d: pm[d]["score"] for d in ("safety", "truthfulness", "consistency") if d in pm}
        ts = trust_by.get(m, {})
        expected["trustscore"][m] = {
            "trust_score": ts.get("trustworthiness_score", pm.get("trust_score")),
            "dimension_scores": ts.get("dimension_scores", dims),
            "weights": dict(DEFAULT_WEIGHTS),
        }

        ci_model = (ci or {}).get("per_model", {}).get(m, {})
        expected["ci"][m] = {
            d: {
                "mean": c.get("mean"),
                "ci_lower": c.get("ci_lower"),
                "ci_upper": c.get("ci_upper"),
            }
            for d, c in ci_model.items()
        }

    for cfg in (ranking or {}).get("configurations", []):
        rnk = cfg.get("ranking", {})
        winner = next((k for k, v in rnk.items() if v == 1), None)
        expected["ranking"]["configurations"].append(
            {"config": cfg.get("config"), "winner": winner}
        )

    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="Directory containing the committed reports.")
    parser.add_argument("--output",
                        default="results/expected_results.json",
                        help="Path where expected_results.json is written.")
    args = parser.parse_args()

    results = Path(args.results)
    scores = _load(results / "scores_report.json")
    ci = _load(results / "ci_report.json")
    ranking = _load(results / "ranking_stability.json")
    trust = _load(results / "trustscore_report.json")

    if scores is None or ci is None or ranking is None or trust is None:
        print("  ✗ Missing one of scores_report.json / ci_report.json / "
              "ranking_stability.json / trustscore_report.json in", results)
        return 1

    expected = build_expected(scores, ci, ranking, trust)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(expected, indent=2, ensure_ascii=False))

    print(f"  -> {out} written")
    print(f"  models: {expected['models']}")
    print("  ranking winners:", {c['config']: c['winner']
                                   for c in expected['ranking']['configurations']})

    # Audit trail
    log_action(
        log_path=Path("experiment/logs/processing_log.jsonl"),
        action="generate_expected_results",
        script="scripts/generate_expected_results.py",
        args={"--results": args.results, "--output": args.output},
        input_paths={"results_dir": results},
        output_paths={"expected": out},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
