.PHONY: help setup run clean clean-all lint audit dashboard eval offlinescore test \
	blinded-prepare blinded-report blinded-heldout-prepare blinded-heldout-report \
	generate-audit

SHELL := /bin/bash
RESULTS := results
MODELS := gemma3:4b,llama3.1:8b
# Interpreter for project tooling. Defaults to the venv so dev/test deps
# (pytest, scipy, requests) resolve; override with PY=python3 if needed.
PY := .venv/bin/python
ifeq ("$(wildcard $(PY))","")
	PY := python3
endif

help:
	@echo "Trustworthiness Evaluation - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make setup                    Install dependencies via uv"
	@echo "  make run                      Run full evaluation pipeline + dashboard" 
	@echo "                                (has interactive annotation gates)"
	@echo "  make eval                     Run evaluation only (no dashboard)"
	@echo "  make dashboard                Launch Streamlit dashboard only"
	@echo "  make offlinescore             Run offline rescoring"
	@echo "  make generate-audit           Rebuild results/audit/all_audit.jsonl from fresh raw outputs"
	@echo "  make clean                    Clean computed results (model dirs, json/txt/png, raw outputs)"
	@echo "  make clean-all                Remove ALL generated/untracked/ignored artifacts, keep git files"
	@echo "  make lint                     Check code quality with ruff"
	@echo "  make audit                    Generate manual audit file"
	@echo "  make test                     Run automated test suite (fails closed)"
	@echo "  make blinded-prepare          Emit per-annotator blinded annotation templates (calibration)"
	@echo "  make blinded-report           Inter-annotator κ + gold-vs-auto comparison"
	@echo "                                (set ANNOTATIONS=\"b1.jsonl b2.jsonl\" DIMENSION=safety)"
	@echo "  make blinded-heldout-prepare  Emit held-out annotation templates"
	@echo "                                (set ANNOTATORS=\"ann1 ann2\")"
	@echo "  make blinded-heldout-report   Final once-only held-out agreement report"

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

generate-audit:
	@echo "Rebuilding results/audit/all_audit.jsonl from fresh raw outputs..."
	$(PY) scripts/generate_audit_samples.py \
		--raw "$(RESULTS)/raw_outputs" \
		--output "$(RESULTS)/audit/all_audit.jsonl" \
		--n-safety 10 --n-truthfulness 10 --n-consistency 10 --seed 42
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

# Full clean: remove every GENERATED / UNTRACKED / IGNORED artifact under
# results/, WITHOUT touching git-tracked files (committed reference data such
# as all_audit.jsonl, annotation CSV/JSONL, blinded templates, manual audit,
# human_annotation_30.csv are preserved). This is a safe "reset to a fresh,
# reproducible state".
#
# Two sources of junk are swept:
#   * untracked files  -> `git clean -fd` (git never touches committed files),
#   * ignored work dirs (blinded_work/, blinded_heldout_work/ — per-annotator
#     templates filled by humans, listed in .gitignore) -> explicit rm -rf.
# Committed files are always preserved; nothing is lost irrecoverably.
clean-all:
	@echo "Removing generated / untracked / ignored artifacts under $(RESULTS)/..."
	@echo "  (git-tracked files are preserved)"
	git clean -fd -- "$(RESULTS)/"
	@echo "  Removing ignored annotation work dirs..."
	rm -rf "$(RESULTS)/blinded_work" "$(RESULTS)/blinded_heldout_work"
	@echo "  Removing empty leftover directories..."
	@find "$(RESULTS)" -type d -empty -delete 2>/dev/null || true
	@echo "Full clean complete"

lint:
	@echo "Checking code quality..."
	ruff check src/ scripts/ app/ --fix
	@echo "Lint complete"

audit:
	@echo "Generating manual audit file..."
	$(PY) scripts/manual_audit_consistency.py
	@echo "Manual audit generated"

# Test target: runs the suite with the venv interpreter by default, so the dev
# dependencies (pytest, scipy, requests) are actually present. Falls back to a
# user-supplied interpreter via PY=... (e.g. PY=python3 make test).
test:
	$(PY) -m pytest -q

# Blinded multi-rater re-annotation workflow.
# Stage 1: emit per-annotator annotation templates from the blinded JSONL.
# Set ANNOTATORS="ann1 ann2".
blinded-prepare:
	@test -n "$(ANNOTATORS)" || (echo "Set ANNOTATORS=\"ann1 ann2\""; exit 1)
	@echo "Preparing blinded annotation templates (calibration)..."
	$(PY) scripts/run_blinded_annotation.py prepare \
		--input results/audit/blinded/blinded_annotation_calibration.jsonl \
		--output results/blinded_work \
		--annotators $(ANNOTATORS)
	@echo "  → Edit results/blinded_work/<annotator>.jsonl, then run 'make blinded-report'"

# Stage 2: inter-annotator agreement + gold-vs-auto comparison.
# Set DIMENSION=safety|truthfulness|consistency and ANNOTATIONS="a1.jsonl a2.jsonl".
blinded-report:
	@test -n "$(ANNOTATIONS)" || (echo "Set ANNOTATIONS=\"ann1.jsonl ann2.jsonl\""; exit 1)
	@test -n "$(DIMENSION)" || (echo "Set DIMENSION=safety|truthfulness|consistency"; exit 1)
	@echo "Computing inter-annotator agreement..."
	$(PY) scripts/run_blinded_annotation.py report \
		--annotations $(ANNOTATIONS) \
		--dimension $(DIMENSION) \
		--audit results/audit/all_audit.jsonl \
		--output results/audit/inter_annotator_report.json

# Held-out stage: the held-out validation set is blinded and annotated exactly
# once, at the very end, to yield the final agreement figures.
# It must NOT be tuned against (unlike the calibration split).
# Set ANNOTATORS="ann1 ann2".
blinded-heldout-prepare:
	@test -n "$(ANNOTATORS)" || (echo "Set ANNOTATORS=\"ann1 ann2\""; exit 1)
	@echo "Preparing blinded held-out annotation templates..."
	$(PY) scripts/run_blinded_annotation.py prepare \
		--input results/audit/blinded/blinded_annotation_heldout.jsonl \
		--output results/blinded_heldout_work \
		--annotators $(ANNOTATORS)
	@echo "  → Edit results/blinded_heldout_work/<annotator>.jsonl, then run 'make blinded-heldout-report'"

# Final once-only held-out report. Aggregates all dimensions. Set ANNOTATIONS
# to the FULL paths of the filled templates, e.g.
# ANNOTATIONS="results/blinded_heldout_work/ann1.jsonl results/blinded_heldout_work/ann2.jsonl".
blinded-heldout-report:
	@test -n "$(ANNOTATIONS)" || (echo "Set ANNOTATIONS to the full template paths"; exit 1)
	@echo "Computing held-out inter-annotator + gold + auto agreement..."
	$(PY) scripts/run_blinded_annotation.py report \
		--annotations $(ANNOTATIONS) \
		--dimension all \
		--audit results/audit/all_audit.jsonl \
		--output results/audit/inter_annotator_report_heldout.json
	@echo "  → Held-out report written to results/audit/inter_annotator_report_heldout.json"

