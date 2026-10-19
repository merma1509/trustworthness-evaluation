# Changelog

All notable changes to the trustworthiness-evaluation pipeline are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

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

- **Added** `TruthfulnessScorer with reject-then-invent fix.\*\*
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
    `verify_immutable` passes (9 artifacts, no drift).

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
