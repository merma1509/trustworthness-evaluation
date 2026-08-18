.PHONY: help setup run clean lint audit dashboard eval offlinescore test \
	blinded-prepare blinded-report

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
	@echo "  make setup        Install dependencies via uv"
	@echo "  make run          Run full evaluation pipeline + dashboard"
	@echo "  make eval         Run evaluation only (no dashboard)"
	@echo "  make dashboard    Launch Streamlit dashboard only"
	@echo "  make offlinescore Run offline rescoring"
	@echo "  make clean        Remove generated results"
	@echo "  make lint         Check code quality with ruff"
	@echo "  make audit        Generate manual audit file"
	@echo "  make test         Run automated test suite (fails closed)"
	@echo "  make blinded-prepare    Emit per-annotator blinded annotation templates"
	@echo "  make blinded-report     Inter-annotator κ + gold-vs-auto comparison"
	@echo "                          (set ANNOTATIONS=\"b1.jsonl b2.jsonl\" DIMENSION=safety)"

setup:
	@echo "Installing dependencies with uv (including dev extras)..."
	uv sync --extra dev
	@echo "Dependencies installed"

run:
	@echo "Running full evaluation pipeline + dashboard..."
	./demo.sh
	@echo "Pipeline complete"

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

