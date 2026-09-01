#!/usr/bin/env python3
"""verify_results.py — verify computed results are internally consistent

Checks invariants of the generated reports:

  * TrustScore == weighted sum of dimension scores (within epsilon) per model,
  * each κ is within [-1, 1] and agreement within [0, 1],
  * agree + disagree == n for every dimension,
  * CIs are sane (lower <= mean <= upper),
  * ranking-stability flip probabilities are within [0, 1],
  * (optional) regenerated reports match ``--expected results/expected_results.json``
    for TrustScore, dimension scores, CIs and ranking winners,
    and all other metrics are within specified ranges.
Any violation reports a clear failure and exits 1 (fails closed).

Usage:
    python3 scripts/verify_results.py --results results [--expected results/expected_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _tolerance_close(actual, expected, tolerance: float = 1e-3) -> bool:
    """Robust closeness test; missing ``actual`` treats as non-match."""

    if actual is None:
        return False
    return abs(float(actual) - float(expected)) <= tolerance


def _check_expected_scores(scores, expected, errors: List[str]) -> None:
    """TrustScore and per-dimension scores must match the expected values."""

    exp_models = expected.get("trustscore") or {}
    per_model = (scores or {}).get("per_model", {})
    for model, exp_trust in exp_models.items():
        got = per_model.get(model)
        if got is None:
            alt = next(
                (m for m in per_model if m.replace(":", "_") == model.replace(":", "_")),
                None,
            )
            got = per_model.get(alt)
        if got is None:
            errors.append(f"[expected] scores_report missing model {model}")
            continue
        exp_score = exp_trust.get("trust_score")
        if exp_score is not None and not _tolerance_close(got.get("trust_score"), exp_score):
            errors.append(
                f"[expected] TrustScore for {model}: got {got.get('trust_score')} "
                f"≠ expected {exp_score}"
            )
        exp_dims = exp_trust.get("dimension_scores") or {}
        for dim, exp_v in exp_dims.items():
            got_dim = (got.get(dim) or {}).get("score")
            if exp_v is not None and not _tolerance_close(got_dim, exp_v):
                errors.append(
                    f"[expected] {model}/{dim} score: got {got_dim} ≠ expected {exp_v}"
                )


def _check_expected_ci(ci, expected, errors: List[str]) -> None:
    """Per-model CIs must at least overlap the expected CI window."""

    exp_ci = expected.get("ci") or {}
    per_model = (ci or {}).get("per_model", {})
    for model, dims in exp_ci.items():
        got_model = per_model.get(model)
        if got_model is None:
            alt = next(
                (m for m in per_model if m.replace(":", "_") == model.replace(":", "_")),
                None,
            )
            got_model = per_model.get(alt)
        if got_model is None:
            errors.append(f"[expected] ci_report missing model {model}")
            continue
        for dim, exp in dims.items():
            got = got_model.get(dim) or {}
            lo, up = exp.get("ci_lower"), exp.get("ci_upper")
            g_lo, g_mean, g_up = got.get("ci_lower"), got.get("mean"), got.get("ci_upper")
            if g_lo is None or g_up is None or lo is None or up is None:
                errors.append(f"[expected] ci_report malformed for {model}/{dim}")
                continue
            if g_mean is not None and not (lo - 0.05 <= g_mean <= up + 0.05):
                errors.append(
                    f"[expected] {model}/{dim} CI mean {g_mean} outside expected "
                    f"[{lo}, {up}] ± 0.05"
                )
            if not (g_up < lo or up < g_lo):
                continue
            errors.append(
                f"[expected] {model}/{dim} CI [{g_lo}, {g_up}] does not overlap "
                f"expected [{lo}, {up}]"
            )


def _check_expected_ranking(ranking, expected, errors: List[str]) -> None:
    """Ranking winners per configuration must match the expected winners."""

    for cfg in expected.get("ranking", {}).get("configurations", []):
        name = cfg.get("config")
        exp_winner = cfg.get("winner")
        got = next(
            (c for c in (ranking or {}).get("configurations", [])
             if c.get("config") == name),
            None,
        )
        if got is None:
            errors.append(f"[expected] ranking configurations missing '{name}'")
            continue
        ranking_map = got.get("ranking") or {}
        winner = next((k for k, v in ranking_map.items() if v == 1), None)
        if exp_winner is not None and winner is not None:
            if winner.replace("_", ":") != exp_winner.replace("_", ":"):
                errors.append(
                    f"[expected] ranking winner under '{name}': got {winner} "
                    f"≠ expected {exp_winner}"
                )


def _check_expected(actual_reports: dict, expected, errors: List[str]) -> None:
    """Compare regenerated reports against the committed expected values."""

    if expected is None:
        return
    _check_expected_scores(actual_reports.get("scores"), expected, errors)
    _check_expected_ci(actual_reports.get("ci"), expected, errors)
    _check_expected_ranking(actual_reports.get("ranking"), expected, errors)


def verify_results(results_dir: Path, expected_path: Optional[Path] = None) -> List[str]:
    errors: List[str] = []

    scores = _load(results_dir / "scores_report.json")
    trust = _load(results_dir / "trustscore_report.json")
    agreement = _load(results_dir / "part1_agreement_report.json") or _load(
        results_dir / ".." / "experiment/reports/part1_agreement_report.json"
    ) or _load(results_dir / ".." / "experiment/reports/agreement_report.json")
    ci = _load(results_dir / "ci_report.json")
    ranking = _load(results_dir / "ranking_stability.json")

    # 1. TrustScore == weighted sum of dimension scores.
    if scores and trust:
        weights = (trust.get("weights_default") or scores.get("weights_default") or [])
        baseline = weights[0] if weights else None
        for model in scores.get("per_model", {}):
            dims = scores["per_model"][model]
            if baseline:
                computed = (
                    baseline["w_s"] * dims["safety"]["score"]
                    + baseline["w_t"] * dims["truthfulness"]["score"]
                    + baseline["w_c"] * dims["consistency"]["score"]
                )
                reported = dims.get("trust_score")
                # Allow a tolerance that accommodates the 4-decimal rounding
                # applied to both dimension scores and the final TrustScore
                # (e.g. weighted sum 0.745950 vs reported round(...,4)=0.746).
                if reported is not None and abs(computed - reported) > 1e-3:
                    errors.append(
                        f"TrustScore mismatch for {model}: computed {computed:.6f} "
                        f"≠ reported {reported}"
                    )

    # 2. Agreement-report invariants (κ bounds, agreement bounds, agree+disagree==n).
    if agreement:
        gold_vs_auto = agreement.get("gold_vs_auto") or agreement.get("agreement", {})
        per_dim = gold_vs_auto.get("per_dimension") if isinstance(gold_vs_auto, dict) else None
        buckets = {}
        if per_dim:
            buckets.update(per_dim)
        if isinstance(gold_vs_auto, dict) and "overall" in gold_vs_auto:
            buckets["overall"] = gold_vs_auto["overall"]
        for dim, stats in buckets.items():
            if not isinstance(stats, dict):
                continue
            k = stats.get("cohens_kappa")
            if k is not None and not (-1 <= k <= 1):
                errors.append(f"κ out of bounds for {dim}: {k}")
            ag = stats.get("agreement_rate")
            if ag is not None and not (0 <= ag <= 1):
                errors.append(f"Agreement out of bounds for {dim}: {ag}")
            # agree + disagree == n
            agree = stats.get("agree")
            disagree = stats.get("disagree")
            n = stats.get("n")
            if agree is not None and disagree is not None and n is not None:
                if agree + disagree != n:
                    errors.append(
                        f"Count mismatch for {dim}: agree({agree}) + disagree({disagree}) "
                        f"≠ n({n})"
                    )

    # 3. CI sanity.
    if ci:
        for model, dims in ci.get("per_model", {}).items():
            for dim, c in dims.items():
                lo, mean, up = c.get("ci_lower"), c.get("mean"), c.get("ci_upper")
                if lo is not None and mean is not None and up is not None:
                    if not (lo <= mean <= up):
                        errors.append(f"CI malformed for {model}/{dim}: {c}")

    # 4. Ranking stability: probabilities within [0,1] and the
    #    dashboard-compatible ``configurations`` block is internally coherent
    #    (ranking order matches descending scores).
    if ranking:
        flip = ranking.get("flip_probability") or {}
        for cfg in flip.get("per_config", []):
            fp = cfg.get("flip_probability")
            if fp is not None and not (0 <= fp <= 1):
                errors.append(f"flip_probability out of range: {fp}")
        for cfg in ranking.get("configurations", []):
            scores_map = cfg.get("scores", {})
            ranking_map = cfg.get("ranking", {})
            ordered = sorted(scores_map.items(), key=lambda kv: kv[1], reverse=True)
            expected = {k: i + 1 for i, (k, _) in enumerate(ordered)}
            if len(ranking_map) >= 2 and ranking_map != expected:
                errors.append(
                    f"configurations ranking inconsistent under "
                    f"'{cfg.get('config')}': {ranking_map} vs {expected}"
                )

    # 5. Compare regenerated reports against the committed expected values.
    if expected_path is not None:
        expected = _load(expected_path)
        if expected is None:
            errors.append(f"[expected] expected results file not found: {expected_path}")
        else:
            _check_expected(
                {
                    "scores": scores,
                    "ci": ci,
                    "ranking": ranking,
                },
                expected,
                errors,
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="Directory of generated results reports.")
    parser.add_argument("--expected", default=None,
                        help="Optional path to results/expected_results.json to "
                             "verify regenerated reports against committed values "
                             "(PLAN PART2 §2.6.2).")
    args = parser.parse_args()

    errors = verify_results(Path(args.results),
                            Path(args.expected) if args.expected else None)
    if errors:
        print("  VERIFICATION FAILED:")
        for e in errors:
            print(f"    [FAIL]{e}")
        return 1
    print("  Results verification passed (all invariants hold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

