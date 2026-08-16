# Peer-Review Revision Plan

**Status:** In progress — Tasks 1–6, 8, 9 substantially complete; Task 7 & 10 partially done (documentation/protocol written; final re-annotation & claim update pending held-out experiment).
**Scope:** Address the external course-reviewer feedback on the Trustworthiness Evaluation study.

---

## 0. Problem Statement

The course reviewer identified that several headline results are **incorrect or overstated**,
and that the human audit is **not clean enough** to validate the automatic scorer. The requested
order of operations is:

> 1. Repair the scorer / statistical pipeline.
> 2. Add automated tests.
> 3. Version **one canonical scoring specification**.
> 4. Re-annotate an **independently blinded multi-rater** sample with **separate calibration and
>    held-out validation** sets.
> 5. **Do not** make deployment recommendations or broad "validated trustworthiness" claims
>    until the re-validation experiment succeeds.

This plan breaks that down into concrete, verifiable tasks.

---

## 1. Known Issues (from reviewer feedback + our verification)

Each row is one concrete defect. The "Fix" column points to the task that repairs it.

| #   | Issue                                                               | Evidence                                                                                                                                                                        | Fix        |
| --- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | Cohen's κ is not the standard Cohen's κ                             | `src/agreement.py` uses `1/n_categories` as chance agreement, valid only for balanced marginals. Corrected estimates ≈ 0.697 / 0.783 / 0.286 / −0.111 (overall / Sa / Tr / Co). | Task 3     |
| 2   | "9.3× fewer failures" is arithmetically wrong                       | 9 vs 3 failures → ratio is **3×**, not 9.3×.                                                                                                                                    | Task 2     |
| 3   | "All TrustScore differences < 0.02" is false                        | Truthfulness-heavy config yields diff **0.0359**.                                                                                                                               | Task 2     |
| 4   | RQ2 attack-type analysis is missing                                 | `validation_report.json`: all 30 audit records have `attack_type="unknown"`.                                                                                                    | Task 5     |
| 5   | Jackknife / sample-size analysis cannot establish required future N | Current method resamples the _existing_ binary scores; it never simulates genuinely new prompts and mixes duplicated prompt texts.                                              | Task 6     |
| 6   | Dataset: 105 records but only 76 unique prompt texts                | Consistency groups duplicate the same base prompt; joint-dataset dedup not handled.                                                                                             | Task 6     |
| 7   | Consistency estimated from 11 groups/model, not "32 prompts"        | Verified: 11 multi-prompt groups per model in `results/raw_outputs/*_consistency.jsonl`.                                                                                        | Tasks 4, 6 |
| 8   | Human audit leaks `auto_label` to the annotator                     | `MANUAL_AUDIT_RUBRIC.md` step 2 says "Check the auto_label"; annotation artifacts include it.                                                                                   | Task 7     |
| 9   | Rubric has contradictory blinding instructions                      | Rubric step 2 ("check auto_label") contradicts step 3 ("do not peek at auto_label").                                                                                            | Task 7     |
| 10  | Some human labels violate the rubric itself                         | Several consistency pairs labelled "consistent" where rubric criteria differ; truthfulness false-positive cases conflict with rubric.                                           | Task 7     |

---

## 2. Target End-State

After this revision we want, at minimum:

- [x] Correct **standard Cohen's κ** (+ weighted κ where appropriate) implemented, tested, and
      reported with confidence intervals.
- [x] All headline arithmetic verified by an automated test that **fails if a claim regresses**.
- [x] `attack_type` reliably propagated through the audit pipeline (RQ2 works).
- [x] A **statistically defensible** dataset-size estimator (not the current shortcut).
- [ ] A **clean, blinded, multi-rater** human validation protocol with calibration + held-out sets.
- [x] A **versioned, single** scoring specification document that the code and the rubric both
      conform to.
- [ ] Deferred or clearly-labelled as _not yet validated_: deployment recommendations and broad
      "validated trustworthiness" claims.

---

## 3. Work Packages

### WP-A: Statistical correctness (scorer repair)

**Objective:** all agreement statistics and headline numbers must be mathematically correct.

#### Task 1 — Audit the current statistics module

- [x] Read and document the exact behaviour of `_cohen_kappa` in `src/agreement.py`.
- [x] Confirm the marginals of the current 30-record audit set (counts per label per model).
- [x] Write out the correct Cohen's κ formula (from marginals) we will implement.

#### Task 2 — Fix arithmetic claims + make them testable

- [x] Correct the "9.3×" claim → recompute from raw counts (expect **3×**).
- [x] Recompute all weight-configuration differences; fix the "Δ < 0.02" claim to the true max (~0.036).
- [x] Represent these expected values **as constants in a test file** so any regression fails.

#### Task 3 — Implement correct Cohen's κ

- [x] Implement standard Cohen's κ in `src/agreement.py`:

  ```bash
  p_o = Σ_k C_kk / N
  p_e = Σ_k (row_k / N) · (col_k / N)
  κ   = (p_o − p_e) / (1 − p_e)
  ```

- [x] Add **weighted κ** for ordered/multi-class labels (safety: correct/incorrect; consistency).
- [x] Add bootstrap CI around κ (resample records, not pairs independently).
- [x] Update reported values; update `MANUAL_AUDIT_RUBRIC.md` interpretation table if needed.

#### Task 4 — Fix consistency group accounting

- [ ] Make the pipeline explicitly count **unique multi-prompt groups** (currently 11/model).
- [ ] Ensure singletons are excluded from every denominator and reported transparently.
- [ ] Add a test asserting group-count invariants (e.g. for the reference dataset = 11/model).

### WP-B: Attack-type propagation (RQ2)

#### Task 5 — Repair `attack_type` loss in the audit pipeline

- [x] Trace where `attack_type` is dropped (likely `build_*_audit_records` / audit sample assembly).
- [x] Ensure `attack_type` is carried into every audit record (safety, truthfulness, consistency).
- [x] Regenerate the audit set and re-run `paradigm_report.py`; confirm RQ2 now has real attack types.
- [x] Add a test that no audit record is emitted with `attack_type` missing/empty/unknown.

### WP-C: Dataset-size estimation (RQ3)

#### Task 6 — Defensible sample-size analysis

- [x] Distinguish **unique prompts** from **records**; use unique texts/canonical IDs as the unit.
- [x] Replace the current CI-width shortcut with a **power / precision** calculation that:
  - [x] treats each **group** (not each record) as the consistency unit,
  - [x] accounts for the observed κ/agreement,
  - [x] reports the N needed to reach a **stated precision** (e.g. CI half-width) rather than a
        fixed <0.10 shortcut.
- [x] Document the assumption that **new prompts are independent** of the current set.
- [x] Add a test that the estimator returns a monotonic, finite N.

### WP-D: Clean blinded human validation

#### Task 7 — Rebuild the annotation protocol

- [x] Fix the rubric: **remove** the instruction to check `auto_label`; produce a blinded JSONL
      that contains **no** auto label / similarity fields (or an unblinded review copy).
- [x] Add a **multi-rater** protocol (≥2 annotators) and compute **inter-annotator agreement**
      (another Cohen's κ / agreement) _before_ comparing to the auto-scorer. _(Protocol documented; actual 2-annotator run pending re-annotation.)_
- [x] Split the sample into **calibration** (used to tune/understand the scorer) and **held-out
      validation** (used only once, at the end, to report final figures). _(Split by unit via script.)_
- [x] Document adjudication for disagreements.
- [x] Add a lint/test that the annotation schema is consistent (no leaked fields where they must
      be hidden).

### WP-E: Single canonical scoring specification

#### Task 8 — Version one canonical spec

- [x] Write `docs/scoring-spec-v2.md` (or similar) that defines, **unambiguously**:
  - [ ] every label with its definition and examples,
  - [ ] the exact matching/classification rules (safety, truthfulness, consistency),
  - [ ] the consistency threshold and group-counting rule,
  - [ ] the TrustScore formula and weight configurations,
  - [ ] the κ statistic definition.
- [ ] **Reference this single file** from both `src/` docstrings and the human rubric, so they
      cannot drift apart.
- [ ] Add a test that reads the spec and checks the code constants (thresholds, weights, counts)
      match it.

### WP-F: Test harness + CI

#### Task 9 — Automated tests

- [x] Unit tests for `src/agreement.py` (κ formula correctness vs. hand-computed / scikit-learn).
- [x] Unit tests for `src/classifiers.py` against labelled examples from the spec.
- [x] Unit tests for `src/stats.py` (CI, jackknife, weight sensitivity) invariants.
- [x] Integration test for `attack_type` propagation (Task 5).
- [x] Test that the **arithmetic constants** (Task 2) and **group counts** (Task 4) hold.
- [x] Wire tests into `make test` and CI (GitHub Actions) — fails closed on any regression.

### WP-G: Reporting & claims hygiene

#### Task 10 — Update results/claims after re-validation

- [ ] Only after WP-D's held-out experiment succeeds:
  - [x] Re-run the full pipeline, regenerate `validation_report.json`. _(Corrected κ & real attack types now reported.)_
  - [x] Update headline numbers in `README.md` / report: corrected κ (0.697), 3× ratio, real max Δ (0.036).
- [ ] Until then: keep the ⚠️ peer-review banner; explicitly mark any claim as _not yet validated_.
- [x] Keep the ⚠️ peer-review banner; deployment/"validated trustworthiness" claims stay labelled as
      pending WP-D held-out validation.

---

## 4. Suggested Execution Order

The reviewer explicitly said to fix the scorer/pipeline **first**, then tests/spec, then
re-annotate. Recommended order:

1. **WP-A (Tasks 1–4)** — repair κ, arithmetic, group accounting. _(Highest priority.)_
2. **WP-B (Task 5)** — restore `attack_type` so RQ2 is analysable.
3. **WP-E (Task 8)** — write the canonical scoring spec _before_ touching the rubric, so the
   rubric change (Task 7) conforms to a fixed target.
4. **WP-F (Task 9)** — add tests as we go (fail-closed).
5. **WP-C (Task 6)** — defensible N estimator.
6. **WP-D (Task 7)** — blinded multi-rater re-annotation last, since it depends on a repaired
   scorer + spec.
7. **WP-G (Task 10)** — update claims only after held-out validation succeeds.

---

## 5. Definition of Done (success criteria)

This revision counts as "done" when **all** of the following hold:

- [ ] Correct standard Cohen's κ + weighted κ are implemented, unit-tested, and reported with CIs.
- [ ] The arithmetic claims test passes (no "9.3×", no false "Δ < 0.02").
- [ ] `attack_type` is present for 100% of audit records; RQ2 renders a real breakdown.
- [ ] Consistency uses group-level counts (11/model) consistently and transparently.
- [ ] The dataset-size estimator is defensible and documented; unique-prompt basis stated.
- [ ] A single versioned scoring spec exists and both code and rubric conform to it.
- [ ] ≥2 blinded annotators produced disjoint calibration/held-out sets; inter-annotator agreement
      reported.
- [ ] Deployment/validated-trustworthiness claims are absent or clearly labelled as pending.
- [ ] `make test` passes; CI is green.

---

## 6. Open Questions / Decisions

- [ ] Should we use **weighted κ** (quadratic) or unweighted for multi-class (consistency) labels?
      → Decide in Task 3.
- [ ] How many annotators can we realistically recruit for the re-annotation?
      → Decide in Task 7.
- [ ] What is the **minimum reportable precision** (CI half-width) for the N estimator?
      → Decide in Task 6.
- [ ] Are we allowed to **expand the dataset** to reach the required N, or must we estimate only?
      → Possibly a second phase after this revision.
- [ ] Should the canonical spec live in `docs/` and be testable, or stay in `MANUAL_AUDIT_RUBRIC.md`?
      → Decide in Task 8.
- [x] Make the pipeline explicitly count **unique multi-prompt groups** (currently 11/model).
- [x] Ensure singletons are excluded from every denominator and reported transparently.
- [x] Add a test asserting group-count invariants (e.g. for the reference dataset = 11/model).
