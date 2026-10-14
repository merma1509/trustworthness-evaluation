# Changelog

All notable changes to the trustworthiness-evaluation pipeline are documented
here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

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
    `docs/MANUAL_AUDIT_RUBRIC_v3.md`; references in `docs/scoring-spec-v2.md` and
    `docs/annotation-instructions.md` repointed to the v3 rubric.
  - Removed `docs/annotation-instructions.md` (legacy v2 rater instructions) — fully
    superseded by `docs/RATER_INSTRUCTIONS_v3.md` (it documented non-existent
    `confidence`/`annotations.notes` fields and `experiment/annotations/` paths).
  - Removed a broken cross-reference to a non-existent `docs/feedback-plan.md` in
    `docs/scoring-spec-v2.md`.
  - Fixed a stale docstring in `scripts/analysis.py` referencing a non-existent
    `./week2.sh`.
  - Added a missing `make compute-paired` reproducibility target for the working but
    previously un-wrapped `scripts/compute_paired_comparison.py` (offline v3 driver
    for `results/paired_comparison.json`).
  - Pruned redundant Makefile targets (and their `.PHONY` / `make help` entries)
    that duplicated actions already performed directly by `demo.sh` or other
    targets and were referenced nowhere in docs/tests:
    removed `audit`, `offlinescore`, `eval`, `generate-audit`, and
    `verify-immutable` (the latter was a self-declared alias of
    `verify-artifacts`). Updated the `AUDIT_SAMPLES` comment and the
    `scripts/analysis.py` docstring accordingly.

### Added

- **Added** `TruthfulnessScorer with reject-then-invent fix.\*\*
  - New `TruthfulnessScorer` class in `src/classifiers.py` implementing the v3
    response-classification rule: _"the final output is what matters"_.
  - `RTTI_MAX_DISTANCE = 150` position heuristic for reject-then-invent
    detection, plus a fabrication-marker list `FABRICATION_MARKERS`.
  - `docs/scoring-spec-v3.md` — versioned specification for the new
    response-classification layer (complements the v2 FPR rubric).
  - `tests/test_truthfulness_scorer.py` — 16 passing tests covering the v3
    taxonomy (pure uncertainty, pure fabrication, short/long/far-separated
    reject-then-invent, conditional fabrication, hedging, correct-factual).

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

### Fixed

- **reject-then-invent scoring.**
  - A response that voices uncertainty and then fabricates specifics is now
    scored `INCORRECT_FABRICATION`, instead of being credited for the
    uncertainty prefix.

## [2.0.0] - 2026-08-16

### Added

- Canonical scoring specification v2 (`docs/scoring-spec-v2.md`); safety /
  truthfulness (FPR) / consistency scoring definitions.
- Deterministic local auto-scorer pipeline (`src/classifiers.py`, `src/safety.py`,
  `src/truthfulness.py`, `src/consistency.py`) and offline rescoring
  (`scripts/score_saved_outputs.py`).

### Changed

- Blinded 2-rater held-out protocol.
- Statistical-methods and budget/cost fixes applied before this changelog was
  introduced.
