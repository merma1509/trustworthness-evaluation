.PHONY: help setup run clean clean-all lint format audit dashboard eval offlinescore test \
	generate-audit experiment-audit experiment-prepare experiment-heldout-prepare \
	experiment-heldout-report experiment-blinded-verify experiment-budget \
	backfill-audit human-timing budget-figure error-heatmap pipeline-figure \
	experiment-seal experiment-annotate experiment-ingest experiment-resolve \
	experiment-gold experiment-agreement experiment-seal-verify experiment-reproduce \
	verify-artifacts compute-scores compute-ci compute-ranking generate-results-json \
	generate-final-manifest verify-immutable verify-results generate-expected


# Interpreter for project tooling. Defaults to the venv so dev/test deps
# (pytest, scipy, requests) resolve; override with PY=python3 if needed
SHELL := /bin/bash
RESULTS := results
MODELS := gemma3:4b,llama3.1:8b
PY := .venv/bin/python
ifeq ("$(wildcard $(PY))","")
	PY := python3
endif


# ── Shared recipe: build an audit dataset (single source of truth) ──
# Used by both `generate-audit` (small sample) and `experiment-audit` (full
# sample). Each caller sets the target-specific variables below:
#   AUDIT_OUTPUT       output jsonl path
#   AUDIT_N_SAFETY     # samples per-dimension (safety)
#   AUDIT_N_TRUTHFULNESS
#   AUDIT_N_CONSISTENCY
define AUDIT_SAMPLES
	$(PY) scripts/generate_audit_samples.py \
		--raw "$(RESULTS)/raw_outputs" \
		--output $(AUDIT_OUTPUT) \
		--n-safety $(AUDIT_N_SAFETY) --n-truthfulness $(AUDIT_N_TRUTHFULNESS) \
		--n-consistency $(AUDIT_N_CONSISTENCY) --seed 42
endef

help:
	@echo "Trustworthiness Measurement Evaluation - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make setup                    Install dependencies via uv"
	@echo "  make run                      Run full evaluation pipeline + dashboard"
	@echo "                                (fresh audit + paper-analysis figures; interactive"
	@echo "                                gates; auto-builds held-out report if templates filled)"
	@echo "  make eval                     Run evaluation only (no dashboard)"
	@echo "  make dashboard                Launch Streamlit dashboard only"
	@echo "  make offlinescore             Run offline rescoring"
	@echo "  make generate-audit           Rebuild results/audit/all_audit.jsonl from fresh raw outputs"
	@echo "  make clean                    Clean computed results (model dirs, json/txt/png, raw outputs)"
	@echo "  make clean-all                Remove ALL generated/untracked/ignored artifacts, keep git files"
	@echo "  make lint                     Check code quality with ruff"
	@echo "  make format                   Auto-format code with ruff (format + import sort)"
	@echo "  make audit                    Generate manual audit file"
	@echo "  make test                     Run automated test suite (fails closed)"
	@echo ""
	@echo "PAPER ANALYSIS (cost / budget figures — also run by 'make run'):"
	@echo "  make experiment-budget REPORT=<json>          Trust-budget plan (κ-gated human allocation)"
	@echo "  make human-timing DIMENSION=safety SAMPLE=8   MEASURED human timing study (interactive)"
	@echo "  make budget-figure KAPPAS=...                 Budget-vs-reliability figure"
	@echo "  make error-heatmap                            Auto×Human error heatmap"
	@echo "  make pipeline-figure                          Measurement-validation loop diagram"
	@echo ""
	@echo "LEGACY blinded 2-rater experiment (full-dataset, anonymised):"
	@echo "  NOTE: superseded by CLEAN-REDO below (3 raters, sealed, label-constrained)"
	@echo "  Kept only while README / committed reports still cite its held-out κ figures"
	@echo "  make experiment-audit                 Regenerate FULL audit dataset for the experiment"
	@echo "  make experiment-prepare               Build anonymised calibration/held-out + ground truth"
	@echo "  make experiment-heldout-prepare       Emit blank annotator templates (ANNOTATORS=\"ann1 ann2\")"
	@echo "  make experiment-heldout-report        Final held-out report (ANNOTATIONS=\"<2+ filled files>\")"
	@echo "  make backfill-audit                   Backfill human labels into all_audit_full.jsonl"
	@echo "  make experiment-blinded-verify        Verify no auto_label/model/prompt_id leaked"
	@echo ""
	@echo "CLEAN-REDO (sealed-randomization protocol) - CANONICAL:"
	@echo "  make experiment-seal                   Seal a NEW experiment (templates + encrypted labels)"
	@echo "                                           SEED=... EXPERIMENT_ID=... [PASSPHRASE=...]"
	@echo "  make experiment-annotate               Emit blank rater templates + declaration forms"
	@echo "                                           RATERS=\"raterA raterB\""
	@echo "  make experiment-ingest ANNOTATIONS=... Schema-validate filled annotations"
	@echo "  make experiment-resolve                Adjudicate A/B disagreements (SPLIT=..., rater files)"
	@echo "  make experiment-gold                   Build gold_labels.jsonl (SEAL_PASSPHRASE=...)"
	@echo "  make experiment-agreement              Held-out agreement figures (gold-vs-auto, inter-rater, k+CI)"
	@echo "  make experiment-seal-verify            Check a sealed experiment is intact & fully blinded"
	@echo "  make experiment-reproduce              ONE command: regenerate EVERY reported number"
	@echo "                                           (scores, CI, ranking, cost, κ) from immutable inputs"
	@echo "                                           then verify internal consistency + artifact checksums"
	@echo ""
	@echo "REPRODUCIBILITY DRIVERS (used by experiment-reproduce; run individually too):"
	@echo "  make compute-scores                    Per-model/per-dimension scores + TrustScore (results/scores_report.json)"
	@echo "  make compute-ci                        Bootstrap CIs n=10000 seed=42 (results/ci_report.json)"
	@echo "  make compute-ranking                   Ranking stability / flip probability (results/ranking_stability.json)"
	@echo "  make generate-results-json             Assemble final result reports (results/trustscore_report.json)"
	@echo "  make generate-final-manifest           Write results/manifest.json (SHA-256 checksums)"
	@echo "  make verify-artifacts                  Verify immutable artifacts match the manifest"
	@echo "  make verify-results                    Verify generated results are internally consistent (vs results/expected_results.json)"
	@echo "  make generate-expected                 Snapshot current reports into results/expected_results.json (committed expectations)"
	@echo "  make verify-immutable                  Check artifact checksums (alias of verify-artifacts)"

setup:
	@echo "Installing dependencies with uv (including dev extras)..."
	uv sync --extra dev
	@echo "Dependencies installed"

run:
	@echo "Running full evaluation pipeline + dashboard..."
	@echo "Note: the pipeline pauses at manual-annotation steps. In a non-interactive"
	@echo "shell, set RUN_INTERACTIVE=0 to skip the annotation gates."
	./demo.sh
	@echo "Pipeline complete"

generate-audit: AUDIT_OUTPUT := "$(RESULTS)/audit/all_audit.jsonl"
generate-audit: AUDIT_N_SAFETY := 10
generate-audit: AUDIT_N_TRUTHFULNESS := 10
generate-audit: AUDIT_N_CONSISTENCY := 10
generate-audit:
	@echo "Rebuilding $(AUDIT_OUTPUT) from fresh raw outputs..."
	$(AUDIT_SAMPLES)
	@echo "Audit dataset regenerated (human labels reset to empty)"

eval:
	@echo "Running evaluation only..."
	$(PY) run_evaluation.py --models $(MODELS) --output $(RESULTS)
	$(PY) scripts/analysis.py
	@echo "Evaluation complete"

offlinescore:
	@echo "Offline rescoring saved outputs..."
	$(PY) scripts/score_saved_outputs.py \
		--input "$(RESULTS)/raw_outputs/*.jsonl" \
		--output "$(RESULTS)/rescored_verification.json" \
		--dimension all
	@echo "Offline rescoring complete"

dashboard:
	@echo "Launching Streamlit dashboard..."
	$(PY) -m streamlit run app/dashboard.py

clean:
	@echo "Cleaning previous results..."
	rm -rf $(RESULTS)/gemma3_4b $(RESULTS)/llama3.1_8b
	rm -f $(RESULTS)/*.json $(RESULTS)/*.txt $(RESULTS)/*.png
	rm -f $(RESULTS)/raw_outputs/*.jsonl
	@echo "Cleaned"


# ========= Full clean: ==========
# remove every GENERATED / UNTRACKED / IGNORED artifact under
# results/, WITHOUT touching git-tracked files (committed reference data such
# as all_audit.jsonl, annotation CSV/JSONL, blinded templates, manual audit,
# human_annotation_30.csv are preserved). This is a safe "reset to a fresh,
# reproducible state"
#
# Two sources of junk are swept:
#   * untracked files  -> `git clean -fd` (git never touches committed files),
#   * ignored work dirs (experiment/held_out_work/ — per-annotator templates
#     filled by humans, listed in .gitignore) -> explicit rm -rf
# Committed files are always preserved; nothing is lost irrecoverably
clean-all:
	@echo "Removing generated / untracked / ignored artifacts..."
	@echo "  (git-tracked files are preserved; calibration_work/ is kept as evidence)"
	@echo "  Removing experiment/audit/ (duplicate of paradigm_report.json)..."
	rm -rf experiment/audit/
	@echo "  Removing results/ untracked..."
	git clean -fd -- "$(RESULTS)/"
	@echo "  Removing results/ ignored work dirs..."
	rm -rf "$(RESULTS)/blinded_work" "$(RESULTS)/blinded_heldout_work"
	@echo "  Removing experiment/ held-out work (per-annotator files, gitignored)..."
	rm -rf experiment/held_out_work
	@echo "  Removing empty leftover directories..."
	@find "$(RESULTS)" -type d -empty -delete 2>/dev/null || true
	@find "experiment" -type d -empty -delete 2>/dev/null || true
	@echo "  Note: experiment/calibration_work/ and experiment/paradigm_report.json are kept."
	@echo "  (untracked but important: show how calibration ties were resolved)"
	@echo "Full clean complete"

lint:
	@echo "Checking code quality..."
	ruff check src/ scripts/ app/ --fix
	@echo "Lint complete"

# Auto-format code with ruff (format + import sorting) then re-check
format:
	@echo "Formatting code..."
	ruff format src/ scripts/ app/
	ruff check --select I src/ scripts/ app/ --fix
	@echo "Formatting complete"

audit:
	@echo "Generating manual audit file..."
	$(PY) scripts/manual_audit_consistency.py
	@echo "Manual audit generated"

# Test target: runs the suite with the venv interpreter by default, so the dev
# dependencies (pytest, scipy, requests) are actually present. Falls back to a
# user-supplied interpreter via PY=... (e.g. PY=python3 make test)
test:
	$(PY) -m pytest -q


# ── blinded held-out EXPERIMENT (full-dataset, anonymised flow) ──
# The experiment flow is the single blinded-annotation flow. It runs on the
# FULL audit dataset with strict anonymisation: Model A/B + sequential anon ids,
# dimension kept (needed for the label rubric) but auto_label/attack_type/
# prompt_id/model hidden. The analyst-only ground truth lets the report
# re-attach auto labels afterwards

# Regenerate the full audit dataset (all prompts/groups) used by the experiment
experiment-audit: AUDIT_OUTPUT := experiment/all_audit_full.jsonl
experiment-audit: AUDIT_N_SAFETY := 100
experiment-audit: AUDIT_N_TRUTHFULNESS := 100
experiment-audit: AUDIT_N_CONSISTENCY := 100
experiment-audit:
	@echo "Regenerating FULL audit dataset for the blinded experiment..."
	$(AUDIT_SAMPLES)
	@echo "  -> $(AUDIT_OUTPUT)"

# Build the anonymised calibration/held-out blinded files + secret ground truth.
experiment-prepare:
	@echo "Building anonymised blinded calibration/held-out datasets (seed=42)..."
	$(PY) scripts/generate_blinded_annotation.py \
		--audit experiment/all_audit_full.jsonl \
		--raw "$(RESULTS)/raw_outputs" \
		--output experiment/blinded \
		--calibration-ratio 0.3 --seed 42
	@echo "  -> experiment/blinded/{blinded_annotation_calibration,blinded_annotation_heldout}.jsonl"
	@echo "  -> experiment/blinded/ground_truth_blinded.json (ANALYST-ONLY, git-ignored)"


# ==== CLEAN-REDO: sealed-randomization protocol ====
# Fresh, label-constrained human-validation pipeline that fixes the reviewer's
# integrity findings (annotation mismatch, prompt/model leakage, silent
# exclusion). Uses the FINITE label vocabulary from src/labels.py and seals the
# ground truth separately from the rater templates
#
#    experiment-seal          seal a NEW experiment (templates + encrypted auto labels)
#    experiment-annotate      emit per-rater blank templates + declaration forms
#    experiment-ingest        schema-validate filled annotations (label-constrained gate)
#    experiment-resolve       adjudicate A/B disagreements -> gold labels
#    experiment-gold          build gold_labels.jsonl (gold + sealed auto join)
#    experiment-agreement     held-out agreement figures (gold-vs-auto, inter-rater)
#    experiment-seal-verify   check a sealed experiment is intact & fully blinded
#    experiment-reproduce     run the FULL pipeline end-to-end with simulated raters
#
# Most steps need variables; run `make help` or see each script's --help

# Seal a new experiment. Pass PASSPHRASE explicitly for reproducibility, or use
# --generate-passphrase via the target below.
experiment-seal:
	@test -n "$(SEED)" || (echo "Set SEED=..."; exit 1)
	$(PY) scripts/seal_experiment.py \
		--raw "$(RESULTS)/raw_outputs" \
		--experiment-dir experiment/sealed \
		--seed "$(SEED)" \
		--experiment-id "$(EXPERIMENT_ID)" \
		$(if $(PASSPHRASE),--passphrase "$(PASSPHRASE)",--generate-passphrase --keeper experiment/.sealed_keeper.json)
	@echo "  Sealed experiment ready. NEXT: make experiment-annotate"

# Emit one blank template + declaration form per rater
experiment-annotate:
	$(PY) scripts/onboard_raters.py \
		--experiment-dir experiment/sealed \
		--out-experiment experiment \
		--raters $(RATERS) ADJUDICATOR

# Schema-validate filled annotation files (label-constrained gate)
# Pass ANNOTATIONS="..." with the filled jsonl paths
experiment-ingest:
	$(PY) scripts/ingest_annotations.py \
		--annotations $(ANNOTATIONS) \
		--manifest experiment/manifests/annotation_manifest.json \
		--declarations-dir experiment/rater_declarations

# Resolve A/B disagreements with the adjudicator -> agreements JSON.
# Pass SPLIT=calibration|heldout and point the rater files via ANNOTATIONS
experiment-resolve:
	$(PY) scripts/resolve_disagreements.py \
		--rater-a $(RATER_A_FILE) \
		--rater-b $(RATER_B_FILE) \
		--adjudicator $(ADJUDICATOR_FILE) \
		--split $(SPLIT) \
		--experiment-id "$(EXPERIMENT_ID)" \
		--out experiment/agreements/$(EXPERIMENT_ID)_$(SPLIT)_disagreements.json

# Build gold_labels.jsonl (join gold human label + sealed auto)
# Requires a recoverable SEAL_PASSPHRASE.
experiment-gold:
	$(PY) scripts/generate_gold_labels.py \
		--resolutions experiment/agreements/$(EXPERIMENT_ID)_heldout_disagreements.json \
		--sealed-labels experiment/sealed/labels/sealed_auto_labels.jsonl.enc \
		--passphrase "$(SEAL_PASSPHRASE)" \
		--out experiment/gold/gold_labels.jsonl

# Final held-out agreement report (gold vs auto + inter-rater A vs B)
experiment-agreement:
	$(PY) scripts/report_part1_agreement.py \
		--gold experiment/gold/gold_labels.jsonl \
		--rater-a $(RATER_A_FILE) \
		--rater-b $(RATER_B_FILE) \
		--out experiment/reports/part1_agreement_report.json \
		--with-ci

# Verify a sealed experiment is intact & fully blinded
experiment-seal-verify:
	$(PY) scripts/verify_seal_integrity.py \
		--manifest experiment/sealed/manifests/sealing_manifest.json \
		--sealed-dir experiment/sealed \
		--sealed-labels experiment/sealed/labels/sealed_auto_labels.jsonl.enc \
		--keys-file experiment/.sealed_keeper.json

# One-command reproduction of the entire pipeline regenerates
# EVERY reported number (scores, CI, ranking stability, cost) from immutable
# raw outputs + sealed artifacts, then verifies internal consistency and
# checks artifact checksums. Deterministic (fixed seeds)
experiment-reproduce: verify-artifacts
	@echo "=== Reproducing all reported numbers ==="
	@echo "--- 1/7 sealed-randomization end-to-end (simulated raters) ---"
	$(PY) scripts/demo_e2e.py
	@echo "--- 2/7 dimension scores + TrustScore ---"
	$(PY) scripts/compute_scores.py --raw "$(RESULTS)/raw_outputs" --output "$(RESULTS)/scores_report.json"
	$(PY) scripts/generate_results_json.py --scores "$(RESULTS)/scores_report.json" --output-results "$(RESULTS)"
	@echo "--- 3/7 bootstrap CIs (n=10000, seed=42) ---"
	$(PY) scripts/compute_ci.py --raw "$(RESULTS)/raw_outputs" --n-bootstrap 10000 --output "$(RESULTS)/ci_report.json"
	@echo "--- 4/7 ranking stability ---"
	$(PY) scripts/compute_ranking_stability.py --scores "$(RESULTS)/scores_report.json" --n-bootstrap 10000 --output "$(RESULTS)/ranking_stability.json"
	@echo "--- 5/7 cost analysis (from committed / measured timing) ---"
	@if [ -f "$(RESULTS)/cost_tracker.json" ]; then \
		echo "  Using existing (MEASURED / pipeline) cost_tracker.json — RQ4 cost is a human-timed value"; \
	else \
		echo "  No cost_tracker.json found; RQ4 cost unavailable (run measure_human_annotation_time.py interactively)."; \
	fi
	@echo "--- 6/7 final manifest ---"
	$(PY) scripts/generate_final_manifest.py --output "$(RESULTS)/manifest.json"
	@echo "--- 7/7 verify ---"
	$(PY) scripts/verify_results.py --results "$(RESULTS)" --expected "$(RESULTS)/expected_results.json"
	$(PY) scripts/verify_immutable.py --manifest "$(RESULTS)/manifest.json" --base-dir .
	@echo "=== All numbers reproduced & verified ==="

# Verify immutable artifacts match their manifest checksums
verify-artifacts:
	@echo "Verifying immutable artifacts..."
	$(PY) scripts/verify_immutable.py --manifest "$(RESULTS)/manifest.json" --base-dir .
	@echo "All artifacts verified."

# Recompute per-model/per-dimension scores + TrustScore from raw outputs
compute-scores:
	$(PY) scripts/compute_scores.py --raw "$(RESULTS)/raw_outputs" --output "$(RESULTS)/scores_report.json"
	$(PY) scripts/generate_results_json.py --scores "$(RESULTS)/scores_report.json" --output-results "$(RESULTS)"

# Bootstrap confidence intervals (n=10000, seed=42)
compute-ci:
	$(PY) scripts/compute_ci.py --raw "$(RESULTS)/raw_outputs" --n-bootstrap 10000 --output "$(RESULTS)/ci_report.json"

# Ranking stability (flip probability) from the scores report
compute-ranking:
	$(PY) scripts/compute_ranking_stability.py --scores "$(RESULTS)/scores_report.json" --n-bootstrap 10000 --output "$(RESULTS)/ranking_stability.json"

# Assemble final result reports (TrustScore report) from score data
generate-results-json:
	$(PY) scripts/generate_results_json.py --scores "$(RESULTS)/scores_report.json" --output-results "$(RESULTS)"

# Write results/manifest.json with SHA-256 checksums of all artifacts
generate-final-manifest:
	$(PY) scripts/generate_final_manifest.py --output "$(RESULTS)/manifest.json"

# Verify generated results are internally consistent
verify-results:
	$(PY) scripts/verify_results.py --results "$(RESULTS)" --expected "$(RESULTS)/expected_results.json"

# Write results/expected_results.json — the committed expectations that
# `verify_results.py --expected` checks regenerated reports against
# Snapshots current scores/CI/ranking values; run
# only when a NEW reference result set is intentionally published.
generate-expected:
	$(PY) scripts/generate_expected_results.py --results "$(RESULTS)" --output "$(RESULTS)/expected_results.json"
	@echo "Expected results snapshotted from current reports into $(RESULTS)/expected_results.json."

# Check artifact checksums against the manifest
verify-immutable:
	$(PY) scripts/verify_immutable.py --manifest "$(RESULTS)/manifest.json" --base-dir .

experiment-heldout-prepare:
	@test -n "$(ANNOTATORS)" || (echo "Set ANNOTATORS=\"ann1 ann2\""; exit 1)
	@echo "Preparing held-out annotation templates (anonymised)..."
	$(PY) scripts/run_blinded_annotation.py prepare \
		--input experiment/blinded/blinded_annotation_heldout.jsonl \
		--output experiment/held_out_work \
		--annotators $(ANNOTATORS)
	@echo "  -> Edit experiment/held_out_work/<annotator>.jsonl, then run 'make experiment-heldout-report'"

# Final once-only held-out report (aggregates all dimensions). Set ANNOTATIONS to
# the FILLED template paths, e.g.,
# ANNOTATIONS="experiment/held_out_work/ann1.jsonl experiment/held_out_work/ann2.jsonl"
experiment-heldout-report:
	@test -n "$(ANNOTATIONS)" || (echo "Set ANNOTATIONS to the filled template paths"; exit 1)
	@echo "Computing held-out inter-annotator + gold + auto agreement..."
	$(PY) scripts/run_blinded_annotation.py report \
		--annotations $(ANNOTATIONS) \
		--dimension all \
		--audit experiment/all_audit_full.jsonl \
		--ground-truth experiment/blinded/ground_truth_blinded.json \
		--output experiment/held_out_agreement_report.json
	@echo "  -> Held-out report written to experiment/held_out_agreement_report.json"


# Backfill human gold labels into all_audit_full.jsonl from experiment reports
# and calibration annotations. After this, paradigm_report.py can read human
# labels directly from the audit file
# Run this AFTER experiment-heldout-report (and after filling calibration ties)
backfill-audit:
	@echo "Backfilling human labels into experiment/all_audit_full.jsonl..."
	$(PY) scripts/backfill_audit.py \
		--audit experiment/all_audit_full.jsonl \
		--ground-truth experiment/blinded/ground_truth_blinded.json \
		--experiment-report experiment/held_out_agreement_report.json \
		--calibration-ann experiment/calibration_work/ann1_calibration.jsonl \
		--calibration-ann experiment/calibration_work/ann2_calibration.jsonl \
		--output experiment/all_audit_full.jsonl
	@echo "  -> experiment/all_audit_full.jsonl updated"

# Verify the blinded files leak nothing (auto_label/model/prompt_id/attack_type)
experiment-blinded-verify:
	@echo "Verifying anonymisation of experiment/blinded..."
	$(PY) -c "import json,glob; \
LEAKED=['auto_label','similarity','human_label','expected_behavior','attack_type','prompt_id','group_id','scorer_label']; \
mods=['gemma3_4b','llama3.1_8b']; \
probs=[]; \
[probs.append((p,k)) for p in glob.glob('experiment/blinded/blinded_annotation_*.jsonl') for l in open(p) if l.strip() for k in LEAKED if k in json.loads(l)]; \
[probs.append((p,m)) for p in glob.glob('experiment/blinded/blinded_annotation_*.jsonl') for l in open(p) if l.strip() for m in mods if m in str(json.loads(l))]; \
print('leaks:', len(probs), probs[:10]); \
raise SystemExit(1 if probs else 0)"
	@echo "  -> OK: no leaked fields in experiment/blinded"


# Single source of truth for the budget/cost ratio: MEASURED per-label human
# annotation time via the live interactive timing study. A human annotator
# labels the sampled records one-by-one and the wall-clock time per decision is
# recorded (written to results/human_timing_measurement.json, which both
# paradigm_report.py (RQ4 cost) and budget_optimizer.py consume
#
# Picks the dimension with DIMENSION=safety|truthfulness|consistency and the
# sample size with SAMPLE=n (default 8). This REQUIRES a human at the keyboard
human-timing:
	$(PY) scripts/measure_human_annotation_time.py \
		--input results/audit/all_audit.jsonl \
		--dimension $(if $(DIMENSION),$(DIMENSION),safety) \
		--sample $(if $(SAMPLE),$(SAMPLE),8) \
		--output results/human_timing_measurement.json
	@echo "  -> MEASURED human timing study written to results/human_timing_measurement.json"

# Trust-budget plan: turn the held-out gold-vs-auto κ into a human-allocation
# policy. REPORT defaults to the experiment report once produced.
# Optional gates: GATE_TRUST / GATE_UNVERIFIED override the 0.7 / 0.4 defaults
experiment-budget:
	@test -n "$(REPORT)" || REPORT=experiment/held_out_agreement_report.json; \
	budget=results/budget_plan.json; \
	if [ ! -f "$$REPORT" ]; then \
		echo "✗ Report not found: $$REPORT"; \
		echo "  Produce it with 'make experiment-heldout-report' first."; \
		exit 1; \
	fi; \
	$(PY) scripts/budget_optimizer.py \
		--report $$REPORT \
		--output $$budget \
		$(if $(GATE_TRUST),--gate-trust $(GATE_TRUST)) \
		$(if $(GATE_UNVERIFIED),--gate-unverified $(GATE_UNVERIFIED))
	@echo "  -> Budget plan written to results/budget_plan.json"


# Budget-vs-reliability figure. Optionally pass REPORT=<json> or
# inline KAPPAS="safety=0.615 truthfulness=0.0 consistency=0.615"
budget-figure:
	@REPORT_ARG=""; \
	if [ -n "$(REPORT)" ] && [ -f "$(REPORT)" ]; then REPORT_ARG="--report $(REPORT)"; fi; \
	KAPPA_ARG=""; \
	if [ -n "$(KAPPAS)" ]; then KAPPA_ARG="--kappas $(KAPPAS)"; fi; \
	$(PY) scripts/budget_reliability_curve.py $$REPORT_ARG $$KAPPA_ARG --output results/budget_reliability_curve.png
	@echo "  -> Figure written to results/budget_reliability_curve.png"

# Auto × Human error heatmap from the agreement/validation reports
error-heatmap:
	$(PY) scripts/error_heatmap.py \
		--agreement results/audit/agreement_report.json \
		--validation results/validation_report.json \
		--output results/error_heatmap.png
	@echo "  -> Figure written to results/error_heatmap.png"

# Measurement-validation loop diagram. Pure structure — no
# numeric values, so it is valid for every run regardless of the fresh data
pipeline-figure:
	$(PY) scripts/pipeline_diagram.py --output results/pipeline_loop.png
	@echo "  -> Figure written to results/pipeline_loop.png"

