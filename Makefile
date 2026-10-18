.PHONY: help setup run clean clean-all lint format dashboard test \
	experiment-audit experiment-budget \
	human-timing budget-figure error-heatmap pipeline-figure cost-report cost-frontier \
	experiment-seal experiment-annotate experiment-adjudicator-prep \
	experiment-ingest experiment-resolve \
	experiment-gold experiment-agreement experiment-seal-verify experiment-reproduce \
	experiment-validate-rerun validate-rerun \
	verify-artifacts compute-scores compute-ci compute-ranking compute-paired generate-results-json \
	generate-final-manifest verify-results generate-expected


# Interpreter (venv by default; override with PY=python3)
SHELL := /bin/bash
RESULTS := results
MODELS := gemma3:4b,llama3.1:8b
PY := .venv/bin/python
ifeq ("$(wildcard $(PY))","")
	PY := python3
endif

# Id of the last-sealed experiment (fallback when EXPERIMENT_ID is unset).
LATEST_EXPERIMENT_ID := $(shell $(PY) -c "import sys; sys.path.insert(0,'.'); from src.experiment import find_latest_experiment_id; print(find_latest_experiment_id() or '')" 2>/dev/null)


# Shared recipe: build an audit dataset (AUDIT_* target-specific vars)
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
	@echo "                                gates; reports sealed-experiment status honestly)"
	@echo "  make dashboard                Launch Streamlit dashboard only"
	@echo "  make clean                    Clean computed results (model dirs, json/txt/png, raw outputs)"
	@echo "  make clean-all                Remove ALL generated/untracked/ignored artifacts, keep git files"
	@echo "  make lint                     Check code quality with ruff"
	@echo "  make format                   Auto-format code with ruff (format + import sort)"
	@echo "  make test                     Run automated test suite (fails closed)"
	@echo ""
	@echo "PAPER ANALYSIS (cost / budget figures — also run by 'make run'):"
	@echo "  make experiment-budget REPORT=<json>          Trust-budget plan (κ-gated human allocation)"
	@echo "  make human-timing DIMENSION=safety SAMPLE=8   MEASURED human timing study (interactive)"
	@echo "  make budget-figure KAPPAS=...                 Budget-vs-reliability figure"
	@echo "  make cost-report DISAGREEMENT=0.1             Cost breakdown (MEASURED/ASSUMED/MODEL)"
	@echo "  make cost-frontier KAPPAS=...                 Cost-reliability frontier (MODEL estimate)"
	@echo "  make error-heatmap                            Auto×Human error heatmap"
	@echo "  make pipeline-figure                          Measurement-validation loop diagram"
	@echo ""
	@echo "(sealed-randomization protocol) - CANONICAL methodology:"
	@echo "  make experiment-audit                  Regenerate FULL audit dataset (experiment/all_audit_full.jsonl)"
	@echo "  make experiment-seal                   Seal a NEW experiment (templates + encrypted labels)"
	@echo "                                           SEED=... [EXPERIMENT_ID=...] [PASSPHRASE=...]"
	@echo "                                           (EXPERIMENT_ID optional — auto-derived from run datetime)"
	@echo "  make experiment-annotate               Emit blank rater templates + declaration forms"
	@echo "                                           RATERS=\"raterA raterB\""
	@echo "  make experiment-adjudicator-prep       Build ADJUDICATOR file from ONLY A/B disagreements"
	@echo "                                           (double-blind; run AFTER A/B, BEFORE adjudicator)"
	@echo "                                           SPLIT=..., RATER_A_FILE=..., RATER_B_FILE=... ADJUDICATOR_FILE=..."
	@echo "  make experiment-ingest ANNOTATIONS=... Schema-validate filled annotations"
	@echo "  make experiment-resolve                Adjudicate A/B disagreements (SPLIT=..., rater files)"
	@echo "  make experiment-gold                   Build gold_labels.jsonl (SEAL_PASSPHRASE=...)"
	@echo "  make experiment-agreement              Held-out agreement figures (gold-vs-auto, inter-rater, k+CI)"
	@echo "  make experiment-seal-verify            Check a sealed experiment is intact & fully blinded"
	@echo "  make experiment-reproduce              ONE command: regenerate EVERY reported number"
	@echo "                                           (scores, CI, ranking, cost, κ) from immutable inputs"
	@echo "                                           then verify internal consistency + artifact checksums"
	@echo "                                           FAILS CLOSED unless real rater annotations are READY"
	@echo "  make experiment-validate-rerun         Clean-rerun validation gate: regenerate every reported"
	@echo "                                           number (incl. gold/agreement κ+CIs) from immutable inputs"
	@echo "                                           + held-out annotations, then verify blindness,"
	@echo "                                           no-silent-drop coverage, gold/auto join and κ bounds."
	@echo "                                           FAILS CLOSED on any broken invariant"
	@echo "                                           SEAL_PASSPHRASE=... EXPERIMENT_ID=... [KEEP_RERUN=1]"
	@echo "  make validate-rerun                   Alias of experiment-validate-rerun (one-command gate)"
	@echo ""
	@echo "REPRODUCIBILITY DRIVERS (used by experiment-reproduce; run individually too):"
	@echo "  make compute-scores                    Per-model/per-dimension scores + TrustScore (results/scores_report.json)"
	@echo "  make compute-ci                        Bootstrap CIs n=10000 seed=42 (results/ci_report.json)"
	@echo "  make compute-ranking                   Ranking stability / flip probability (results/ranking_stability.json)"
	@echo "  make compute-paired                    Paired/clustered model comparison CI (results/paired_comparison.json)"
	@echo "  make generate-results-json             Assemble final result reports (results/trustscore_report.json)"
	@echo "  make generate-final-manifest           Write results/manifest.json (SHA-256 checksums)"
	@echo "  make verify-artifacts                  Verify immutable artifacts match the manifest"
	@echo "  make verify-results                    Verify generated results are internally consistent (vs results/expected_results.json)"
	@echo "  make generate-expected                 Snapshot current reports into results/expected_results.json (committed expectations)"

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

dashboard:
	@echo "Launching Streamlit dashboard..."
	$(PY) -m streamlit run app/dashboard.py

clean:
	@echo "Cleaning previous results..."
	rm -rf $(RESULTS)/gemma3_4b $(RESULTS)/llama3.1_8b
	rm -f $(RESULTS)/*.json $(RESULTS)/*.txt $(RESULTS)/*.png
	rm -f $(RESULTS)/raw_outputs/*.jsonl
	@echo "Cleaned"


# Full clean: remove all generated/untracked/ignored artifacts (git files kept)
clean-all:
	@echo "Removing generated / untracked / ignored artifacts..."
	@echo "  (git-tracked files are preserved; sealed reference data is kept)"
	@echo "  Removing results/ untracked..."
	git clean -fd -- "$(RESULTS)/"
	@echo "  Removing experiment/ annotations (per-rater filled files, gitignored)..."
	rm -rf experiment/annotations
	@echo "  Removing empty leftover directories..."
	@find "$(RESULTS)" -type d -empty -delete 2>/dev/null || true
	@find "experiment" -type d -empty -delete 2>/dev/null || true
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

# Run test suite (PY=... to override interpreter)
test:
	$(PY) -m pytest -q


# Full audit dataset (.jsonl) for the sealed experiment
experiment-audit: AUDIT_OUTPUT := experiment/all_audit_full.jsonl
experiment-audit: AUDIT_N_SAFETY := 100
experiment-audit: AUDIT_N_TRUTHFULNESS := 100
experiment-audit: AUDIT_N_CONSISTENCY := 100
experiment-audit:
	@echo "Regenerating FULL audit dataset for the sealed experiment..."
	$(AUDIT_SAMPLES)
	@echo "  -> $(AUDIT_OUTPUT)"


# ==== CLEAN-REDO: sealed-randomization protocol ====

# Seal a new experiment (EXPERIMENT_ID optional: auto-derived from run datetime)
experiment-seal:
	@test -n "$(SEED)" || (echo "Set SEED=..."; exit 1)
	$(PY) scripts/seal_experiment.py \
		--raw "$(RESULTS)/raw_outputs" \
		--experiment-dir experiment/sealed \
		--seed "$(SEED)" \
		$(if $(EXPERIMENT_ID),--experiment-id "$(EXPERIMENT_ID)",) \
		$(if $(PASSPHRASE),--passphrase "$(PASSPHRASE)",--generate-passphrase --keeper experiment/.sealed_keeper.json)
	@echo "  Sealed experiment ready. NEXT: make experiment-annotate"

# Emit blank rater templates + declaration forms
experiment-annotate:
	$(PY) scripts/onboard_raters.py \
		--experiment-dir experiment/sealed \
		--out-experiment experiment \
		--raters $(RATERS) ADJUDICATOR

# Build the ADJUDICATOR file from ONLY the A/B disagreement records
# (double-blind: rebuilt from the sealed template, never from A/B files).
# Run AFTER RATER_A and RATER_B are fully annotated, BEFORE ADJUDICATOR starts.
#     SPLIT=heldout RATER_A_FILE=... RATER_B_FILE=... ADJUDICATOR_FILE=...
experiment-adjudicator-prep:
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	if [ -z "$$EXP_ID" ]; then echo "  ✗ No experiment id found (set EXPERIMENT_ID=... or seal one first)."; exit 1; fi; \
	$(PY) scripts/prepare_adjudicator.py \
		--rater-a $(RATER_A_FILE) \
		--rater-b $(RATER_B_FILE) \
		--sealed "experiment/sealed/templates/$${EXP_ID}_$(SPLIT)_template.jsonl" \
		--out $(ADJUDICATOR_FILE) \
		--experiment-id "$$EXP_ID" \
		--split $(SPLIT)
	@echo "  Adjudicator file prepared (only A≠B disagreements, double-blind)."

# Schema-validate filled annotations (ANNOTATIONS="...")
experiment-ingest:
	$(PY) scripts/ingest_annotations.py \
		--annotations $(ANNOTATIONS) \
		--manifest experiment/manifests/annotation_manifest.json \
		--declarations-dir experiment/rater_declarations

# Adjudicate A/B disagreements (SPLIT=calibration|heldout)
experiment-resolve:
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	$(PY) scripts/resolve_disagreements.py \
		--rater-a $(RATER_A_FILE) \
		--rater-b $(RATER_B_FILE) \
		--adjudicator $(ADJUDICATOR_FILE) \
		--split $(SPLIT) \
		--experiment-id "$$EXP_ID" \
		--out experiment/agreements/$${EXP_ID}_$(SPLIT)_disagreements.json

# Build gold_labels.jsonl
# Reproducible from git: uses the committed plaintext auto labels by default
# (experiment/sealed/labels/auto_labels.jsonl). The sealed path (+passphrase)
# remains for a fresh run that has not yet unsealed auto labels.
experiment-gold:
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	if [ -f experiment/sealed/labels/auto_labels.jsonl ]; then \
		echo "  [gold] using plaintext auto labels (committed, reproducible)"; \
		$(PY) scripts/generate_gold_labels.py \
			--resolutions experiment/agreements/$${EXP_ID}_heldout_disagreements.json \
			--auto-labels experiment/sealed/labels/auto_labels.jsonl \
			--out experiment/gold/gold_labels.jsonl; \
	else \
		echo "  [gold] plaintext auto labels not found; using sealed (+passphrase)"; \
		$(PY) scripts/generate_gold_labels.py \
			--resolutions experiment/agreements/$${EXP_ID}_heldout_disagreements.json \
			--sealed-labels experiment/sealed/labels/sealed_auto_labels.jsonl.enc \
			--passphrase "$(SEAL_PASSPHRASE)" \
			--out experiment/gold/gold_labels.jsonl; \
	fi

# Held-out agreement report (gold vs auto + inter-rater)
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

# Regenerate every reported number from immutable inputs; fails closed unless
# real rater annotations are READY (never fabricates agreement κ)
experiment-reproduce: verify-artifacts
	@echo "=== Reproducing all reported numbers ==="
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	if [ -z "$$EXP_ID" ]; then echo "  ✗ No experiment id found (set EXPERIMENT_ID=... or seal one first)."; exit 1; fi; \
	echo "=== Reproducing all reported numbers (experiment: $$EXP_ID) ==="; \
	echo "--- 0/7 annotation honesty gate (fail-closed) ---"; \
	STATUS=$$($(PY) scripts/experiment_status.py --annotations-dir experiment/annotations --experiment-id "$$EXP_ID" --status-only); \
	echo "  Experiment annotation STATUS: $$STATUS"; \
	if [ "$$STATUS" != "READY" ]; then \
		echo "  ✗ Annotations not complete — refusing to fabricate agreement κ from simulated/short data."; \
		echo "    Fill the sealed rater templates (calibration + heldout) with REAL independent"; \
		echo "    raters, then re-run (see PART 1: experiment-annotate / experiment-ingest / experiment-resolve)."; \
		exit 1; \
	fi
	@echo "--- 1/7 rater agreements -> gold labels -> held-out agreement κ ---"
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	if [ -f "experiment/agreements/$${EXP_ID}_heldout_disagreements.json" ]; then \
		if [ -f "experiment/sealed/labels/auto_labels.jsonl" ]; then \
			echo "    [gold] plaintext auto labels (reproducible from git)"; \
			$(PY) scripts/generate_gold_labels.py \
				--resolutions "experiment/agreements/$${EXP_ID}_heldout_disagreements.json" \
				--auto-labels experiment/sealed/labels/auto_labels.jsonl \
				--out experiment/gold/gold_labels.jsonl; \
		else \
			$(PY) scripts/generate_gold_labels.py \
				--resolutions "experiment/agreements/$${EXP_ID}_heldout_disagreements.json" \
				--sealed-labels experiment/sealed/labels/sealed_auto_labels.jsonl.enc \
				--passphrase "$(SEAL_PASSPHRASE)" \
				--out experiment/gold/gold_labels.jsonl; \
		fi; \
		$(PY) scripts/report_part1_agreement.py \
			--gold experiment/gold/gold_labels.jsonl \
			--rater-a "experiment/annotations/$${EXP_ID}_RATER_A_heldout.jsonl" \
			--rater-b "experiment/annotations/$${EXP_ID}_RATER_B_heldout.jsonl" \
			--out experiment/reports/part1_agreement_report.json \
			--with-ci; \
	else \
		echo "  note: no held-out disagreement-resolution file found; held-out κ not regenerated."; \
		echo "    Run experiment-annotate / experiment-ingest / experiment-resolve with the filled"; \
		echo "    rater files to produce the gold + held-out κ (gold-vs-auto, inter-rater)."; \
	fi
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

# Clean-rerun validation gate: regenerate every number (incl. gold/κ+CIs) from
# immutable inputs + filled annotations; fails closed on any broken invariant
experiment-validate-rerun:
	@echo "=== Clean-rerun VALIDATION gate ==="
	@EXP_ID="$(EXPERIMENT_ID)"; \
	if [ -z "$$EXP_ID" ]; then EXP_ID="$(LATEST_EXPERIMENT_ID)"; fi; \
	if [ -z "$$EXP_ID" ]; then echo "  ✗ No experiment id found (set EXPERIMENT_ID=... or seal one first)."; exit 1; fi; \
	if [ -f experiment/sealed/labels/auto_labels.jsonl ]; then \
		echo "  [validate] using committed plaintext auto labels (no passphrase needed)"; \
		$(PY) scripts/validate_rerun_pipeline.py \
			--experiment-id "$$EXP_ID" \
			--annotations-dir experiment/annotations \
			$(if $(KEEP_RERUN),--keep); \
	else \
		test -n "$(SEAL_PASSPHRASE)" || { \
			echo "  ✗ SEAL_PASSPHRASE=... is required to unseal the auto labels."; \
			echo "    (Or commit experiment/sealed/labels/auto_labels.jsonl.)"; \
			exit 1; \
		}; \
		$(PY) scripts/validate_rerun_pipeline.py \
			--experiment-id "$$EXP_ID" \
			--annotations-dir experiment/annotations \
			--passphrase "$(SEAL_PASSPHRASE)" \
			$(if $(KEEP_RERUN),--keep); \
	fi
	@echo "  validate-rerun complete"

# Alias of experiment-validate-rerun
validate-rerun: experiment-validate-rerun

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

# Paired/clustered model comparison CI (offline v3 path)
compute-paired:
	$(PY) scripts/compute_paired_comparison.py --raw "$(RESULTS)/raw_outputs" --output "$(RESULTS)/paired_comparison.json" --seed 42

# Assemble final result reports (TrustScore report) from score data
generate-results-json:
	$(PY) scripts/generate_results_json.py --scores "$(RESULTS)/scores_report.json" --output-results "$(RESULTS)"

# Write results/manifest.json with SHA-256 checksums of all artifacts
generate-final-manifest:
	$(PY) scripts/generate_final_manifest.py --output "$(RESULTS)/manifest.json"

# Verify generated results are internally consistent
verify-results:
	$(PY) scripts/verify_results.py --results "$(RESULTS)" --expected "$(RESULTS)/expected_results.json"

# Snapshot current reports into results/expected_results.json (committed expectations)
generate-expected:
	$(PY) scripts/generate_expected_results.py --results "$(RESULTS)" --output "$(RESULTS)/expected_results.json"
	@echo "Expected results snapshotted from current reports into $(RESULTS)/expected_results.json."


# MEASURED human timing study (interactive; DIMENSION=..., SAMPLE=n)
human-timing:
	$(PY) scripts/measure_human_annotation_time.py \
		--input results/audit/all_audit.jsonl \
		--dimension $(if $(DIMENSION),$(DIMENSION),safety) \
		--sample $(if $(SAMPLE),$(SAMPLE),8) \
		--output results/human_timing_measurement.json
	@echo "  -> MEASURED human timing study written to results/human_timing_measurement.json"

# Trust-budget plan from held-out κ (GATE_TRUST / GATE_UNVERIFIED optional)
experiment-budget:
	@test -n "$(REPORT)" || REPORT=experiment/reports/part1_agreement_report.json; \
	budget=results/budget_plan.json; \
	if [ ! -f "$$REPORT" ]; then \
		echo "✗ Report not found: $$REPORT"; \
		echo "  Produce it with 'make experiment-agreement' first."; \
		exit 1; \
	fi; \
	$(PY) scripts/budget_optimizer.py \
		--report $$REPORT \
		--output $$budget \
		$(if $(GATE_TRUST),--gate-trust $(GATE_TRUST)) \
		$(if $(GATE_UNVERIFIED),--gate-unverified $(GATE_UNVERIFIED))
	@echo "  -> Budget plan written to results/budget_plan.json"


# Budget-vs-reliability figure (REPORT=<json> or KAPPAS="...")
budget-figure:
	@REPORT_ARG=""; \
	if [ -n "$(REPORT)" ] && [ -f "$(REPORT)" ]; then REPORT_ARG="--report $(REPORT)"; fi; \
	KAPPA_ARG=""; \
	if [ -n "$(KAPPAS)" ]; then KAPPA_ARG="--kappas $(KAPPAS)"; fi; \
	$(PY) scripts/budget_reliability_curve.py $$REPORT_ARG $$KAPPA_ARG --output results/budget_reliability_curve.png
	@echo "  -> Figure written to results/budget_reliability_curve.png"

# Cost breakdown (MEASURED/ASSUMED/MODEL; DISAGREEMENT=0.1 optional)
cost-report:
	$(PY) scripts/compute_cost.py $(if $(DISAGREEMENT),--disagreement $(DISAGREEMENT)) --output results/cost_breakdown.json
	@echo "  -> results/cost_breakdown.json written (MEASURED/ASSUMED/MODEL breakdown)"

# Cost-reliability frontier (MODEL estimate; REPORT=<json> or KAPPAS="...")
cost-frontier:
	@REPORT_ARG=""; \
	if [ -n "$(REPORT)" ] && [ -f "$(REPORT)" ]; then REPORT_ARG="--report $(REPORT)"; fi; \
	KAPPA_ARG=""; \
	if [ -n "$(KAPPAS)" ]; then KAPPA_ARG="--kappas $(KAPPAS)"; fi; \
	$(PY) scripts/cost_reliability_frontier.py $$REPORT_ARG $$KAPPA_ARG --output results/cost_reliability_frontier.json
	@echo "  -> results/cost_reliability_frontier.json written (MODEL estimate)"

# Auto × Human error heatmap from agreement/validation reports
error-heatmap:
	$(PY) scripts/error_heatmap.py \
		--agreement results/audit/agreement_report.json \
		--validation results/validation_report.json \
		--output results/error_heatmap.png
	@echo "  -> Figure written to results/error_heatmap.png"

# Measurement-validation loop diagram (pure structure)
pipeline-figure:
	$(PY) scripts/pipeline_diagram.py --output results/pipeline_loop.png
	@echo "  -> Figure written to results/pipeline_loop.png"

