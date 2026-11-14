# Changelog

All notable changes to the trustworthiness-evaluation pipeline are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed

- **Blinding breach: `metadata.display_model` was shipped to raters.**
  - The seal correctly recorded which model backs each consistency display under
    `metadata.display_model`, but that field was written into the very files
    handed to annotators: all 22 held-out / 42 calibration consistency records
    named `gemma3:4b` / `llama3.1:8b` to raters meant to be blind to it. Model
    identity is a plausible cue for perceived truthfulness, so this was a real
    breach of the blinded protocol, not a cosmetic leak.
  - Rater-visible annotation files now drop the field;
    `experiment/audit/display_model_map.jsonl` keeps
    `internal_key -> display_model` so the audit trail survives the removal.
  - `make experiment-agreement` and `experiment-validate-rerun` are unaffected —
    the model identity they need comes from the sealed auto labels, not from the
    annotation files.

- **The published consistency figures were unverifiable by the rerun gate.**
  - `experiment-validate-rerun` passed held-out annotations only, so it
    regenerated **26/26** repeated verdicts while the write-up quoted **73/74**:
    the gate validated a different report than the one shipped. Both the gate and
    `make experiment-agreement` now receive the calibration split as well.
  - `report_agreement.py::_load_auto_labels` guessed the sealed-label path
    relative to the gold file. A rerun that writes gold to a temporary directory
    made the lookup fail and **silently emptied** the display-level and
    test-retest blocks (`n_displays: 0`, `n_unmapped: 19`) instead of raising.
    The path is now explicit (`--auto-labels`), and when the labels are genuinely
    absent the blocks are emitted as `{"available": false, "reason": ...}` so a
    consumer can tell "not measured" apart from "measured and zero".
  - `_check_agreement` now fails closed when either block is missing, empty or
    marked unavailable, so this class of mismatch cannot pass the gate again.

- **Adjudicator records could carry an unjustified verdict.**
  - `cross_check_adjudicator` proved _which_ records reached the adjudicator but
    never that a verdict was reached. K_0070 entered as
    `INCORRECT_FABRICATION`/`uncertain`, had both fields blanked, and reappeared
    as `CORRECT_UNCERTAINTY`/`confident` — the opposite verdict, presented as
    confident, with no explanation.
  - New `_check_adjudicator_integrity` fails closed unless every adjudicated
    record carries a dimension label and a `rater_uncertainty`, and unless a
    dispute where **both** raters abstained carries an adjudication note (there
    the adjudicator is the only decision-maker).

- **Inter-rater confusion matrices were read transposed.**
  - `compute_agreement(human, auto)` indexes the matrix `{auto: {human: ...}}`,
    and the inter-rater block passes RATER_B as `auto`, so the stored matrix is
    `{rater_b: {rater_a: ...}}` while it declared `rows: rater_a`. Every consumer
    read it transposed, inverting who was the stricter rater. The declared axes
    now match the data layout.

- **Dead CLI flags that silently accepted input and did nothing.**
  - `--inter-rater-calibration` was declared on `report_agreement.py` and read by
    nothing, so passing it produced identical output: a caller could believe
    calibration agreement had been measured when it had not. It now emits a
    separate `inter_rater_calibration_A_vs_B` block, kept apart from
    `inter_rater_A_vs_B` so calibration records can never dilute the held-out
    reliability figure.
  - `--templates` was likewise declared and unused, and its help text was
    **factually wrong** (it claimed display identity could only be recovered from
    the template, whereas it comes from the sealed auto labels' `group_id`).
    Removed rather than documented, since a flag that does nothing is worse than
    no flag.
  - `tests/test_inter_rater_calibration.py` pins both directions: the flag must
    change the output, and the held-out block must stay byte-identical either way.

### Added

- **`kappa_interpretable` marks a kappa that cannot express reliability.**
  - Slices shipped `"cohens_kappa": 0.0` where `p_observed == p_expected`
    (truthfulness: agreement 0.8947, kappa 0.0). The number is arithmetically
    correct — `p_expected` is 0.8947, so the denominator is 0.1053, not zero —
    but read alone it says "chance level", which is false.
  - A `null` would have been the wrong fix (the coefficient _is_ defined), so the
    value stays and `kappa_interpretable` / `kappa_degenerate_reason` make the
    caveat machine-readable. Two distinct degeneracies are distinguished:
    `undefined_pe_zero` (kappa is `None`; single observed class) and
    `observed_equals_expected` (defined but carrying no information).
  - `tests/test_kappa_interpretability.py` covers both degeneracies plus a
    genuine-agreement slice that must stay interpretable.

- **`experiment/audit/display_model_map.jsonl`** — the de-blinded audit trail for
  consistency displays (`internal_key -> display_model`), so removing the field
  from rater-visible files does not also destroy the record of which model was
  shown.

### Changed

- **`docs/report.md` is no longer tracked.**
  - `.gitignore` already listed it under _"Regenerated reports … do not commit"_,
    but the file had been force-added earlier, so the ignore never applied. It is
    now untracked (`git rm --cached`) to match the documented policy; the file
    remains on disk as a local artifact.

- **`make experiment-agreement` now passes `--auto-labels` and
  `--inter-rater-calibration`**, so the canonical report includes the
  display-level, test-retest and calibration blocks that the write-up quotes.

- **Lint gate green again.** Six ruff violations in the restored
  `report_agreement.py` (missing `Optional` import, ambiguous loop name, unused
  local, trailing whitespace, placeholder-free f-string) are fixed; `make lint`
  and the CI `Lint (ruff)` job pass.

## [2.1.0] - 2026-11-12

### Changed

- **`staging` branch retired — `develop` promotes directly to `main`.**
  - The QA role of `staging` was redundant: `develop → main` pull requests run
    the identical quality gate (pytest, ruff, `validate-rerun`,
    `verify-artifacts`), so the extra hop added a branch to keep in sync
    without adding a check. `staging` held no commit that was not already on
    `develop`.
  - `.github/workflows/pr.yml` now accepts `develop → main` as the release
    promotion; `hotfix/*` may still target `main`. The `staging` case was removed
    from the branch-policy guard.
  - `.github/workflows/ci.yml` and `pr.yml` no longer trigger on `staging`.
  - `CONTRIBUTING.md` §1 and §6, `SECURITY.md`, and `cd.yml` updated to the
    two-long-lived-branch model.
  - Stale promotion PRs (#25 `develop → main`, #26 `staging → main`) were
    closed.

- **Sealed templates now embed a rubric + self-referential template checksum.**
  - `scripts/seal_experiment.py` fills `metadata.rubric_checksum` (SHA-256 of
    `docs/MANUAL_AUDIT_RUBRIC_v3.md` at seal time) and
    `metadata.template_checksum` (acyclic SHA-256 of the record with that field
    blanked) on every sealed template record, so the rubric revision handed to
    raters and the exact template each rater annotated are provable from the
    record itself — closing the "rubric checksums were empty" gap (plan-01 §4).
  - `tests/test_clean_rerun_safeguards.py` locks the new contract: checksums are
    non-empty, the template checksum is self-referential (recomputable with the
    field blanked), and the rubric checksum propagates verbatim.

- **Repository moved to a GitFlow branching model (3 long-lived branches).**
  - `main` — production (default branch, protected: PR-only, no force push /
    no deletion).
  - `staging` — testing / QA (protected: PR + review, no force push).
  - `develop` — development / draft (protected: PR-only, no force push).
  - Feature / fix / docs branches are now created off `develop` and promoted
    through `develop → staging → main` via pull requests.
  - Old one-phase branches (`clean-rerun/validation-v2`,
    `feat/sealed-randomization-repro`, `fix/truthfulness-scorer-rtti`,
    `part1-correctness-fixes`, `v2.0-fixes`) were deleted locally and on the
    remote — all of them were ancestors of `main` (their tip commits are
    reachable from `origin/main`), so no work was lost.
  - Added release tags: **`v2.0.0`** (Annotated, stable release) on `main`;
    missing pre-release tags (`v2.0-pre-provenance`,
    `v2.0-results-2026-08-08`) were also pushed to the remote.
  - Added project governance docs: `CONTRIBUTING.md` (GitFlow workflow, branch
    naming, commit conventions, reproducibility gates) and `SECURITY.md`
    (supported versions, private vulnerability reporting, data-handling rules),
    plus a shared pull-request template (`.github/PULL_REQUEST_TEMPLATE.md`).

- **Held-out gold / agreement now reproducible from git without a passphrase.**
  - Added committed plaintext auto labels
    (`experiment/sealed/labels/auto_labels.jsonl`, deterministic canonical
    serialisation, 210 records) alongside the encrypted sealed copy. A
    `.gitignore` negation now keeps this single plaintext file tracked while
    `sealed_auto_labels.jsonl.enc` and the keeper stay private.
  - `scripts/generate_gold_labels.py` now accepts the plaintext auto labels as a
    JSON array **or** JSONL (`--auto-labels`), matching both the sealed payload
    and the on-disk committed file. The sealed `--sealed-labels + --passphrase`
    path is preserved as the fallback.
  - `scripts/validate_rerun_pipeline.py` no longer hard-requires
    `--passphrase` when the committed plaintext auto labels exist: it builds
    gold from `experiment/sealed/labels/auto_labels.jsonl` directly and only
    falls back to the sealed path otherwise.
  - `Makefile` targets `experiment-gold`, `experiment-reproduce` (step 1/7) and
    `experiment-validate-rerun` now prefer the plaintext path and only require
    `SEAL_PASSPHRASE` when no plaintext is present.
  - Result: from a clean git clone, `make experiment-validate-rerun` regenerates
    every number (gold labels, gold-vs-auto κ+CIs, scores/TrustScore, CIs,
    ranking stability) from immutable inputs with **no local secrets**.

- **Repository hygiene: removed legacy / superseded / duplicated files.**
  - Removed `requirements.txt` — legacy dependency list superseded by
    `pyproject.toml` + `uv.lock` (canonical via `make setup` = `uv sync --extra dev`).
    Its contents (`torch`, `transformers`, `tqdm`, `nltk`, `httpx`, …) were never
    imported anywhere in the codebase.
  - Removed unused dependencies from `pyproject.toml`: `ollama>=0.6.2` (only used as
    external CLI via `subprocess`, not a Python import) and `seaborn>=0.12.0` (never
    imported). `uv.lock` re-resolved accordingly.
  - Removed dead constants from `src/labels.py` (`EXPERIMENT_ID_DEFAULT`,
    `UNCERTAINTY_LEVELS`, `RATER_IDS`) that were not referenced anywhere.
  - Removed `data/raw/` (4 seed files) — `data/final/*.jsonl` are a strict superset
    (verified: every raw `prompt_id` is present in final), and no code/Makefile/docs
    read `data/raw`. README updated to not mention the seeds.
  - Removed `docs/MANUAL_AUDIT_RUBRIC.md` (v2 rubric) — fully superseded by
    `docs/MANUAL_AUDIT_RUBRIC_v3.md`; references in `docs/scoring-spec.md` and
    `docs/annotation-instructions.md` repointed to the v3 rubric.
  - Removed `docs/annotation-instructions.md` (legacy v2 rater instructions) — fully
    superseded by `docs/RATER_INSTRUCTIONS_v3.md` (it documented non-existent
    `confidence`/`annotations.notes` fields and `experiment/annotations/` paths).
  - Removed a broken cross-reference to a non-existent `docs/feedback-plan.md` in
    `docs/scoring-spec.md`.
  - Fixed a stale docstring in `scripts/analysis.py` referencing a non-existent
    `./week2.sh`.
  - Added a missing `make compute-paired` reproducibility target for the working but
    previously un-wrapped `scripts/compute_paired_comparison.py` (offline v3 driver
    for `results/paired_comparison.json`).
  - Merged the scoring spec into a single canonical file: `docs/scoring-spec-v2.md`
    (canonical safety / truthfulness / consistency + TrustScore + kappa) and
    `docs/scoring-spec-v3.md` (reject-then-invent response-classification layer)
    are now one document, `docs/scoring-spec.md` (v3, with the v2 core intact as
    sections 1–6 and the v3 truthfulness layer as section 7). The two old files
    were removed; references in `tests/test_scoring_spec.py`,
    `tests/test_classifiers.py`, `src/stats.py` and the changelog were repointed
    to `docs/scoring-spec.md`.
  - Pruned redundant Makefile targets (and their `.PHONY` / `make help` entries)
    that duplicated actions already performed directly by `demo.sh` or other
    targets and were referenced nowhere in docs/tests:
    removed `audit`, `offlinescore`, `eval`, `generate-audit`, and
    `verify-immutable` (the latter was a self-declared alias of
    `verify-artifacts`). Updated the `AUDIT_SAMPLES` comment and the
    `scripts/analysis.py` docstring accordingly.

- **Dashboard tabs now map to the project's research questions (RQ1–RQ4).**
  - New `RQ2 — What Affects Agreement` tab (`app/tabs/rq2_factors.py`) replaces the
    old `Failure Analysis` tab: it shows agreement factors (dimension / attack type /
    model) straight from `results/audit/agreement_report.json`, with a read-out of
    the worst cell. `app/tabs/failure_analysis.py` was removed (superseded by
    `rq2_factors.py`).
  - Each remaining tab carries a small caption stating its role in the project
    (results layer vs. a specific RQ); the `Research Questions` tab now explains
    which sections answer RQ1–RQ4.
  - Footer reworded from a course-project tagline to a neutral research label.
  - `Rater Annotation` tab (`app/tabs/rater_annotation.py`) now shows the rater's
    declaration-form status (signed / blank / missing) in a read-only block
    placed before annotation.
  - `Rater Annotation` tab shows a handoff note when the `ADJUDICATOR` role is
    selected: the adjudicator file is the disagreements-only output of
    `make experiment-adjudicator-prep`, not the full-size template of
    `make experiment-annotate`. The generic "No annotation template found" hint
    is no longer misleading for the adjudicator.
  - `README.md` gains a "Dashboard tabs → research questions" mapping;
    `docs/appendix.md` adds §A.11 with the same mapping.
  - Renamed the `RQ2 — What Affects Agreement` tab to `What Affects Agreement`
    (the `RQ2` prefix was redundant — the tab's role in the appendix already
    states it answers RQ2; stripped `(CLEAN-REDO)` from the
    `Sealed Multi-Rater Re-Validation` heading in `app/tabs/inter_annotator.py`).
    Appendix/README mapping updated to match.

- **Human-label timing is now a single source of truth.**
  - `scripts/budget_reliability_curve.py` previously hard-coded
    `HUMAN_SECONDS_PER_LABEL = 8.01`, drifting from the MEASURED 12.2 s median
    in `results/human_timing_measurement.json` and from every other cost
    artifact (`budget_plan.json`, `cost_breakdown.json`,
    `cost_reliability_frontier.py`). It now imports the constant from
    `scripts/budget_optimizer` (which resolves it from the timing file at import
    time), so the curve, the budget plan and the cost breakdown can never diverge
    again.
  - `scripts/cost_reliability_frontier.py` gets the same single-source import
    (previously a literal 12.2 / 20.0). Values are unchanged (12.2 s, \$20/h);
    only the provenance is now explicit and shared.
  - `README.md` now carries an explicit human-timing caveat: the measured 12.2 s
    comes from a _small, single-dimension_ interactive study (n = 8, safety) with
    high variance (std 73.9 s, one 224.7 s outlier) and is extrapolated to all
    three dimensions — so the budget is indicative, not precise.

### Added

- **Added** `TruthfulnessScorer` with reject-then-invent fix.
  - New `TruthfulnessScorer` class in `src/classifiers.py` implementing the v3
    response-classification rule: _"the final output is what matters"_.
  - `RTTI_MAX_DISTANCE = 150` position heuristic for reject-then-invent
    detection, plus a fabrication-marker list `FABRICATION_MARKERS`.
  - `docs/scoring-spec.md` — merged canonical specification (v2 + v3) defining
    the FPR rubric and the v3 response-classification layer (reject-then-invent).
  - `tests/test_truthfulness_scorer.py` — 16 passing tests covering the v3
    taxonomy (pure uncertainty, pure fabrication, short/long/far-separated
    reject-then-invent, conditional fabrication, hedging, correct-factual).

- **Experiment id is now auto-derived from the run's UTC date/time.**
  - New `src/experiment.py`: `generate_experiment_id(now)` builds
    `trustworthiness-validation-YYYY-MM-DD-HHMMSS` from the run timestamp, and
    `find_latest_experiment_id()` reads the current sealed experiment id from
    `experiment/sealed/manifests/sealing_manifest.json`.
  - `scripts/seal_experiment.py` no longer falls back to a random hex uuid when
    `--experiment-id` is omitted; it now generates a date-based id, so every
    fresh seal produces a uniquely-named, readable, sortable experiment.
  - `scripts/experiment_status.py`, `scripts/validate_rerun_pipeline.py`, and
    the `Makefile` consumers (`experiment-seal`, `experiment-resolve`,
    `experiment-gold`, `experiment-reproduce`, `experiment-validate-rerun`)
    now fall back to the _last-sealed_ experiment id via `find_latest_experiment_id()`
    instead of a hard-coded `2025-08-redo`. Explicit `EXPERIMENT_ID=...` still
    overrides the default for reproducible re-seals.
  - `tests/test_experiment.py` — 10 tests covering date-based id generation
    (UTC normalisation, sortability, uniqueness) and manifest lookup
    (missing/malformed manifest handling).

- **Added** one-command clean-rerun validation gate (`make validate-rerun`).
  - New `scripts/validate_rerun_pipeline.py` regenerates EVERY reported number
    (held-out gold labels, gold-vs-auto and inter-rater κ+CIs, scores/TrustScore,
    bootstrap CIs, ranking stability) from the immutable inputs
    (`results/raw_outputs` + sealed templates + filled rater annotations), then
    verifies: seal/blinding integrity, no-silent-drop coverage,
    gold/auto join completeness, κ/agreement-rate/CI bounds and internal
    consistency. Fails CLOSED on any broken invariant; it never fabricates
    agreement figures from simulated raters.
  - `Makefile` targets `experiment-validate-rerun` and `validate-rerun` alias
    (with `SEAL_PASSPHRASE=...`, `EXPERIMENT_ID=...`, optional `KEEP_RERUN=1`).
  - `tests/test_clean_rerun_safeguards.py` — extended with validation-gate
    tests (coverage fail-closed, κ/CI bounds fail-closed, gold/auto join
    completeness).

- **Added** `docs/RATER_INSTRUCTIONS_v3.md` — authoritative, label-constrained
  instructions for raters (RATER_A/RATER_B/ADJUDICATOR) matching the actual
  sealed-template v3.1 schema (finite label sets, `flags.rater_uncertainty`,
  `flags.ai_assistance_used`, no-skip rules, fill-only-fields checklist).
  This supersedes the legacy v2 instructions in `docs/annotation-instructions.md`,
  which referenced non-existent `confidence`/`annotations.notes` fields.

- **Added** disagreements-only, double-blind ADJUDICATOR handoff
  (`make experiment-adjudicator-prep`).
  - New `scripts/prepare_adjudicator.py` builds the ADJUDICATOR file from ONLY the
    records where RATER*A and RATER_B cast different \_valid* votes, rebuilt from
    the _sealed_ template (prompt/response only — no A/B label, no auto_label, no
    model/prompt/attack ids), so the adjudicator never sees how A or B voted.
    An `uncertain` rater does not cast a valid vote (mirrors
    `scripts/resolve_disagreements.py`); a disagreement key missing from the sealed
    template fails closed instead of silently dropping the record.
  - `Makefile` target `experiment-adjudicator-prep` runs AFTER RATER_A/RATER_B are
    fully annotated and BEFORE ADJUDICATOR starts
    (`SPLIT=... RATER_A_FILE=... RATER_B_FILE=... ADJUDICATOR_FILE=...`), falling
    back to the last-sealed experiment id and logging the handoff to
    `experiment/logs/annotation_log.jsonl`.
  - `prepare_adjudicator.py` **deletes any pre-existing ADJUDICATOR file before
    writing** the fresh disagreements-only one, so the full-size template left
    by `make experiment-annotate` can never linger and be shown to the
    adjudicator alongside (or instead of) the real disagreements.
  - `tests/test_prepare_adjudicator.py` — 7 tests covering disagreements-only
    selection, double-blind rebuild, fail-closed coverage, and the
    delete-then-write replacement of a stale full-size file.

### Fixed

- **`diagnose_model_scoping()` silently passed every display.**
  - The check computed `n_mixed` but never incremented it, so "a consistency
    display mixes two models" always reported `0` regardless of the data — a
    validation that could never fail. Each anonymous `pairs[]` entry is now
    resolved back to its source model through the raw outputs and compared with
    `metadata.display_model`, so the check is evidence-based. Result on the
    current seal: 64 displays, 0 mixed, 0 mismatched, 0 unresolved.
  - `tests/test_model_scoping_check.py` pins it down with cases that must FAIL
    (mixed models, display/metadata mismatch, untraceable pair).

- **Legitimately empty ADJUDICATOR held-out file rejected by the gate.**
  - `validate_rerun_pipeline.py` required all three held-out files to be
    non-empty. After the adjudicator-scoping fix the adjudicator receives ONLY
    the A≠B records, so when A and B agree everywhere the file is correctly
    empty (heldout: 0 disagreements of 63). The guard was demanding fabricated
    adjudications. It now requires records only for RATER_A/RATER_B; coverage is
    still enforced downstream by `resolve_disagreements` (63 = 51 scored +
    12 unresolvable).

- **210 rater notes blanked by the re-bucket commit.**
  - Commit `f142958` rewrote the annotation files and cleared
    `metadata.notes` on every record (210/210 present before, 0/210 after).
    Restored all 210 by `internal_key`; the diff is exactly 210 insertions /
    210 deletions and touches only the `notes` field, so no label, flag,
    checksum or timestamp changes and no score is affected.

- **`make experiment-adjudicator-prep SPLIT=...` failed out of the box.**
  - The target had no defaults for `RATER_A_FILE` / `RATER_B_FILE` /
    `ADJUDICATOR_FILE`, so the documented invocation died on
    "expected one argument". Defaults are now derived from the experiment id
    and split, with existence checks and a clear error naming the missing file.

- **reject-then-invent scoring.**
  - A response that voices uncertainty and then fabricates specifics is now
    scored `INCORRECT_FABRICATION`, instead of being credited for the
    uncertainty prefix.

- **Stale `verify_results.py` fallback and out-of-date artifact manifest.**
  - `scripts/verify_results.py` still referenced the legacy
    `experiment/reports/part1_agreement_report.json` name as a fallback; the
    canonical report is `experiment/reports/agreement_report.json` and the dead
    alias is removed.
  - `results/manifest.json` referenced the non-existent
    `part1_agreement_report.json` and pinned stale hashes for regenerated
    outputs (e.g. `ranking_stability.json`), so `make verify-artifacts` failed.
    `scripts/generate_final_manifest.py` now records
    `experiment/reports/agreement_report.json` in its default artifact set, and
    the manifest was regenerated from the current deterministic outputs —
    `verify_immutable` passes (8 artifacts, no drift). The private encrypted
    sealed labels (`sealed_auto_labels.jsonl.enc`) are intentionally excluded
    from the manifest: they are gitignored (keeper passphrase) and unavailable
    in a clean clone, so `verify_immutable` can run from git; their integrity
    is covered by `verify_seal_integrity.py` inside `make
experiment-validate-rerun`.

## [2.0.0] - 2026-08-16

### Added

- Canonical scoring specification v2 (`docs/scoring-spec.md`); safety /
  truthfulness (FPR) / consistency scoring definitions.
- Deterministic local auto-scorer pipeline (`src/classifiers.py`, `src/safety.py`,
  `src/truthfulness.py`, `src/consistency.py`) and offline rescoring
  (`scripts/score_saved_outputs.py`).

### Changed

- Blinded 2-rater held-out protocol.
- Statistical-methods and budget/cost fixes applied before this changelog was
  introduced.
