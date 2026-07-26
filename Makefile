.PHONY: help setup run clean lint audit dashboard eval offlinescore

SHELL := /bin/bash
RESULTS := results
MODELS := gemma3:4b,llama3.1:8b

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

setup:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Dependencies installed"

run:
	@echo "Running full evaluation pipeline + dashboard..."
	./demo.sh
	@echo "Pipeline complete"

eval:
	@echo "Running evaluation only..."
	python3 run_evaluation.py --models $(MODELS) --output $(RESULTS)
	python3 scripts/analysis.py
	@echo "Evaluation complete"

offlinescore:
	@echo "Offline rescoring saved outputs..."
	python3 scripts/score_saved_outputs.py \
		--input "$(RESULTS)/raw_outputs/*.jsonl" \
		--output "$(RESULTS)/rescored_verification.json" \
		--dimension all
	@echo "Offline rescoring complete"

dashboard:
	@echo "Launching Streamlit dashboard..."
	python3 -m streamlit run app/dashboard.py

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
	python3 scripts/manual_audit_consistency.py
	@echo "Manual audit generated"
