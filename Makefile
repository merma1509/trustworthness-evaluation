.PHONY: help setup run clean lint audit

SHELL := /bin/bash
RESULTS := results
MODELS := gemma3:4b,llama3.1:8b

help:
	@echo "Trustworthiness Evaluation - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make setup      Install dependencies via uv"
	@echo "  make run        Run full evaluation pipeline"
	@echo "  make clean      Remove generated results"
	@echo "  make lint       Check code quality with ruff"
	@echo "  make audit      Generate manual audit file"

setup:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Dependencies installed"

run:
	@echo "Running evaluation pipeline..."
	./demo.sh
	@echo "Pipeline complete"

clean:
	@echo "Cleaning previous results..."
	rm -rf $(RESULTS)/gemma3_4b $(RESULTS)/llama3.1_8b
	rm -f $(RESULTS)/*.json $(RESULTS)/*.txt $(RESULTS)/*.png
	rm -f $(RESULTS)/raw_outputs/*.jsonl
	@echo "Cleaned"

lint:
	@echo "Checking code quality..."
	ruff check src/ scripts/ --fix
	@echo "Lint complete"

audit:
	@echo "Generating manual audit file..."
	python3 scripts/manual_audit_consistency.py
	@echo "Manual audit generated"
