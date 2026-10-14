# Deep-Scan Response to Reviewer — Human-Validation Experiment

**Status:** Final (deep scan complete; experiment rerun **blocked** pending real raters)
**Date:** 2025-10-14
**Branch:** `clean-rerun/validation-v2`
**Scope:** Resolves the reviewer's claims about the human-validation experiment
(data integrity, calibration, held-out exclusion, label vocabulary) and records
what is already fixed in code versus what remains a manual/experimental action.

---

## 1. Executive summary

Every reviewer claim was **verified against the repository** and is accurate for
the _quarantined_ legacy experiment. The methodological fixes the reviewer
demanded are **already implemented and test-protected**; the experiment itself
has **not** been re-run, so no new validation numbers exist yet.

| Reviewer claim                                 | Verdict                    | Where the evidence lives                                                        |
| ---------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------- |
| Labels don't correspond to dimension           | **CONFIRMED (legacy run)** | `experiment.tar.gz/held_out_work/ann1.jsonl` — 27 violations                    |
| Calibration κ=0.6377, n=50 not reconstructible | **CONFIRMED (legacy run)** | `experiment.tar.gz/calibration_work/*calibration.jsonl` — 4 records only        |
| 38 held-out cases silently excluded            | **CONFIRMED (legacy run)** | `experiment.tar.gz/held_out_agreement_report.json` → `n_needs_adjudication: 38` |
| Fixed-label vocabulary required                | **Done in code**           | `src/labels.py`, `docs/LABEL_INTERFACE.md`                                      |
| No-skip / no silent-drop accounting required   | **Done in code**           | `scripts/resolve_disagreements.py`, validation gate                             |
| Blindness of sealed templates required         | **Done in code**           | `scripts/seal_experiment.py` + `tests/test_clean_rerun_safeguards.py`           |
| Honest status gate (no fabricated reports)     | **Done in code**           | `scripts/experiment_status.py` → `PENDING`                                      |

---

## 2. Confirmed findings (quarantined legacy run in `experiment.tar.gz/`)

> **Naming caveat:** `experiment.tar.gz` is a _directory_, not an archive. It is
> the quarantine area for the broken legacy experiment. Its misleading name is
> noted in `docs/CHANGELOG.md`.

### 2.1 Labels do not correspond to dimension (27 violations, `ann1`)

Counter of `(dimension, label)` tuples in `experiment.tar.gz/held_out_work/ann1.jsonl`:

| Dimension      | Legacy label | Count |
| -------------- | ------------ | ----- |
| `consistency`  | `correct`    | 11    |
| `truthfulness` | `consistent` | 8     |
| `safety`       | `consistent` | 7     |
| `consistency`  | `incorrect`  | 1     |

These use a legacy vocabulary (`correct` / `incorrect` / `consistent`) that does
not fit the per-dimension semantic (e.g. `consistent` is not a valid _safety_
or _truthfulness_ verdict). The constrained vocabulary is now enforced — see §3.

### 2.2 Calibration figures are not reconstructible

- Legacy report (`held_out_agreement_report.json`) claims **n=50, κ=0.6377,
  agreement=92%** for calibration.
- The actual calibration files (`calibration_work/ann1_calibration.jsonl`,
  `ann2_calibration.jsonl`) contain **4 records each**, in perfect agreement
  (κ=1.0 from those records).
- Conclusion: the published n/κ cannot be reproduced from the stored files.

### 2.3 38 held-out cases excluded

- `held_out_agreement_report.json` → `overall.adjudicated.n_needs_adjudication: 38`.
- These cases were dropped from the gold-vs-auto comparison (auto_comparison
  n values were 38/34/8 against per-dimension totals of 52/46/20), i.e. 38 hard
  cases never reached the gold set — exactly the reviewer's number.

### 2.4 Current annotation files are empty templates

All six files in `experiment/annotations/` are seal-time templates
(0/147 and 0/63 labels filled). Verified live:

```
scripts/experiment_status.py --annotations-dir experiment/annotations
→ EXPERIMENT ANNOTATION STATUS: PENDING
→ 147 unfilled × 3 (calibration), 63 unfilled × 3 (held-out)
→ Held-out A-vs-B: 0 disagreements
```

The gate is **fail-closed** in two complementary ways:

- `scripts/experiment_status.py --fail-pending` exits non-zero whenever the
  status is not `READY`;
- the `make experiment-reproduce` harness runs
  `experiment_status.py --status-only` and aborts the whole pipeline when the
  status token is not `READY`.

Either way a `PENDING` state can never be mistaken for a validated result, and
no agreement κ is ever regenerated from simulated/short data.

---

## 3. What the reviewer demanded — already implemented

| Fix                                     | Implementation                                                                                                                                           | Tests                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| reject-then-invent (truthfulness)       | `TruthfulnessScorer` in `src/classifiers.py` (`RTTI_MAX_DISTANCE=150`, `FABRICATION_MARKERS`)                                                            | `tests/test_truthfulness_scorer.py` (16)                       |
| Bootstrap p-values, not point estimates | `paired_bootstrap_test` in `src/stats.py` (residual-recentring bootstrap, Efron & Tibshirani ch. 16)                                                     | `tests/test_stats.py`                                          |
| Trial-level ranking stability           | `compute_ranking_stability_with_trials` in `src/stats.py`; `results/ranking_stability.json` reports `basis: per-unit trial bootstrap`                    | `tests/test_stats.py`                                          |
| Constrained label vocabulary            | `src/labels.py` — `COMPLY/REFUSE`, `CORRECT_UNCERTAINTY/INCORRECT_FABRICATION`, `CONSISTENT/INCONSISTENT`; `docs/LABEL_INTERFACE.md`                     | `tests/test_labels*.py`                                        |
| No-skip adjudication                    | `scripts/resolve_disagreements.py` — invariant `scored + unresolvable + excluded == recorded`                                                            | `tests/test_clean_rerun_safeguards.py`                         |
| One-command rerun validation gate       | `scripts/validate_rerun_pipeline.py` + `make validate-rerun` — fails closed on coverage/κ/CI/gold-join breakage                                          | `tests/test_clean_rerun_safeguards.py` (validation-gate block) |
| Blindness of sealed templates           | `_assert_template_blind` / `_template_record` in `scripts/seal_experiment.py`                                                                            | `tests/test_clean_rerun_safeguards.py` (blindness block)       |
| Honest budget figures                   | `scripts/compute_cost.py`, `scripts/budget_reliability_curve.py` (explicitly **SIMULATED**), `scripts/cost_reliability_frontier.py` (**MODEL ESTIMATE**) | `tests/test_compute_cost.py`                                   |
| Cost ratio not overclaimed              | `docs/appendix.md` §A.6 — `~108× nominal` is documented as an **apples-to-oranges** ratio, not a validated saving                                        | `tests/test_compute_cost.py` (`test_nominal_ratio_has_caveat`) |

**Test status:** full suite `171 passed`; targeted suites
(`test_truthfulness_scorer.py`, `test_clean_rerun_safeguards.py`,
`test_agreement.py`) — `42 passed`.

---

## 4. Sample-size / RQ3 readiness (reviewer-adjacent)

The RQ3 sample-size machinery already exists and is wired into
`src/validation.py` (used by the validation report):

- `estimate_required_sample_size` — theoretical Wilson/Wald N estimator
  (`src/stats.py`; Appendix A section in `docs/appendix.md`).
- `compute_required_n_empirically` — bootstrap-based empirical N sweep
  (`src/stats.py`, used by `src/validation.py`).
- Both are covered by `tests/test_stats.py`
  (`test_sample_size_estimator_*`, `test_empirical_n_returns_estimate`,
  `test_wilson_and_wald_reference_values`).

These run on whatever held-out annotation data is produced; today they consume
the _current_ scores. When the experiment is re-run, the same drivers apply to
the new gold labels without code changes.

---

## 5. Outstanding work — the blocker

The only genuinely un-resolvable-in-code item is the **experiment rerun itself**:

1. Recruit independent raters (see `scripts/onboard_raters.py` +
   `docs/RATER_INSTRUCTIONS_v3.md` + `docs/MANUAL_AUDIT_RUBRIC_v3.md`).
2. Fill `experiment/annotations/*.jsonl` from `experiment/sealed/templates/`
   (constrained label sets; no-skip; no AI assistance — declarations in
   `experiment/rater_declarations/`).
3. Unseal with `SEAL_PASSPHRASE=...` and run the gate:
   `make experiment-reproduce` and `make validate-rerun`.
4. Only then does a fresh (non-quarantined) agreement/κ/gold report exist, to
   replace the legacy figures the reviewer rejected.

Until then, `experiment_status.py` honestly reports `PENDING` and all
fabrication paths are closed.

---

## 6. Answer to the reviewer, in one line

> **Infrastructure: resolved and test-protected. Experiment: not yet re-run —
> the legacy report is quarantined, its numbers are not reconstructible, and the
> pipeline now fails closed until real rater annotations are provided.**
