#!/usr/bin/env bash
# ============================================================
# Trustworthiness Evaluation — Full Pipeline
# Runs the FULL pipeline including evaluation, analysis, and
# output generation for all deliverables
# ============================================================
set -e  # Exit on any error

# --- Configuration ---
MODELS="gemma3:4b,llama3.1:8b"
RESULTS_DIR="results"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Trustworthiness Evaluation — Full Pipeline${NC}"
echo -e "${BLUE}  Date: $(date)${NC}"
echo -e "${BLUE}  Models: ${MODELS}${NC}"
echo -e "${BLUE}  Output: ${RESULTS_DIR}/${NC}"
echo -e "${BLUE}============================================================${NC}"

# PHASE 1: SETUP
# --- Step 1: Check prerequisites ---
echo -e "\n${YELLOW}[1/10] Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}  Python3 found${NC}"

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}  Ollama found${NC}"

for model in $(echo $MODELS | tr ',' ' '); do
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo -e "${GREEN}  Model '$model' available${NC}"
    else
        echo -e "${YELLOW} Model '$model' not found. Pulling...${NC}"
        ollama pull "$model"
    fi
done

# --- Step 2: Verify datasets ---
echo -e "\n${YELLOW}[2/10] Verifying datasets...${NC}"

REQUIRED_FILES=(
    "data/raw/safety_seeds.jsonl"
    "data/raw/truthfulness_seeds.jsonl"
    "data/raw/consistency_seeds.jsonl"
    "data/raw/benign_controls.jsonl"
    "data/final/safety.jsonl"
    "data/final/truthfulness.jsonl"
    "data/final/consistency.jsonl"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}  $file${NC}"
    else
        echo -e "${YELLOW} $file not found. Regenerating datasets...${NC}"
        python3 scripts/restructure_datasets.py
        break
    fi
done

# --- Step 3: Validate raw seeds ---
echo -e "\n${YELLOW}[3/10] Validating raw seeds...${NC}"
python3 scripts/validate_raw_seeds.py 2>&1 | tail -5
echo -e "${GREEN}  Seed validation complete${NC}"

# --- Step 4: Clear Python cache for fresh imports ---
echo -e "\n${YELLOW}[4/10] Clearing Python cache...${NC}"
find . -path "*/__pycache__/*" -delete 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}  Cache cleared${NC}"

# PHASE 2: EVALUATION
# --- Step 5: Clean previous results ---
echo -e "\n${YELLOW}[5/10] Cleaning previous results...${NC}"

if [ -L "results_latest" ]; then
    rm results_latest
fi

rm -rf "${RESULTS_DIR}/gemma3_4b" "${RESULTS_DIR}/llama3.1_8b"
rm -f "${RESULTS_DIR}/"*.json "${RESULTS_DIR}/"*.txt "${RESULTS_DIR}/"*.png
rm -f "${RESULTS_DIR}/raw_outputs/"*.jsonl
echo -e "${GREEN}  Previous results cleaned${NC}"

# --- Step 6: Run evaluation pipeline ---
echo -e "\n${YELLOW}[6/10] Running evaluation pipeline...${NC}"
echo -e "  Models: ${MODELS}"
echo -e "  Output: ${RESULTS_DIR}/"

START_TIME=$(date +%s)
python3 run_evaluation.py \
    --models "$MODELS" \
    --output "$RESULTS_DIR"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}  Evaluation complete in ${DURATION}s${NC}"

# --- Step 7: Save run summary ---
echo -e "\n${YELLOW}[7/10] Saving run summary...${NC}"

cat > "${RESULTS_DIR}/pipeline_summary.txt" << EOF
=================================================
  Trustworthiness Evaluation — Pipeline Summary
=================================================
Date:        $(date)
Model 1:     $(echo $MODELS | cut -d',' -f1)
Model 2:     $(echo $MODELS | cut -d',' -f2)
Duration:    ${DURATION}s
Seed files:  4 (safety, truthfulness, consistency, benign)
Output dir:  ${RESULTS_DIR}

Reproduction command:
  python3 run_evaluation.py --models ${MODELS} --output ${RESULTS_DIR}

Last run:    $(date +"%Y-%m-%d %H:%M:%S")
===================================================
EOF

echo -e "${GREEN}  Summary saved to ${RESULTS_DIR}/pipeline_summary.txt${NC}"

# PHASE 3: ANALYSIS & VALIDATION
# --- Step 8: Run baseline model validation ---
echo -e "\n${YELLOW}[8/10] Running model validation...${NC}"
python3 scripts/validate_models.py --all-models 2>&1 | tail -15
echo -e "${GREEN}  Model validation complete${NC}"

# --- Step 8.5: Generate manual audit file ---
echo -e "\n${YELLOW}[8.5/10] Generating manual audit file...${NC}"
python3 scripts/manual_audit_consistency.py
echo -e "${GREEN}  Manual audit file saved to results/manual_audit_consistency.jsonl${NC}"
echo -e "${YELLOW} REMINDER: Edit the file and fill in 'human_label'!${NC}"

# --- Step 9: Generate all analysis plots and summary ---
echo -e "\n${YELLOW}[9/10] Generating analysis plots and summary...${NC}"
python3 scripts/analysis.py
echo -e "${GREEN}  Analysis complete${NC}"

# --- Step 10: Run offline rescoring verification ---
echo -e "\n${YELLOW}[10/10] Running offline rescoring verification...${NC}"
python3 scripts/score_saved_outputs.py \
    --input "${RESULTS_DIR}/raw_outputs/*.jsonl" \
    --output "${RESULTS_DIR}/rescored_verification.json" \
    --dimension all 2>&1 | tail -5
echo -e "${GREEN}  Offline rescoring complete${NC}"

# RESULTS SUMMARY
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}  Pipeline complete!${NC}"
echo -e "${GREEN}  Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}  Generated files:${NC}"
echo -e "    results/results_summary.json             — Combined results"
echo -e "    results/gemma3_4b/scores.json            — Gemma dimension scores + confusion matrix"
echo -e "    results/llama3.1_8b/scores.json          — Llama dimension scores + confusion matrix"
echo -e "    results/raw_outputs/*.jsonl              — Full raw model outputs"
echo -e "    results/ci_error_bars.png                — CI bar chart"
echo -e "    results/ranking_flip_boundary.png        — Weight sensitivity plot"
echo -e "    results/score_difference_heatmap.png     — Ranking regions"
echo -e "    results/analysis_summary.txt             — Text summary with all tables"
echo -e "    results/baseline_validation.json         — Cross-model validation"
echo -e "    results/manual_audit_consistency.jsonl   — Manual audit file"
echo -e "    results/rescored_verification.json       — Offline rescoring proof"
echo ""
echo -e "${YELLOW}  NOTE: To complete manual audit:${NC}"
echo -e "    Edit results/manual_audit_consistency.jsonl"
echo -e "    Change 'human_label': null -> 'consistent' or 'inconsistent'"
echo -e "${BLUE}============================================================${NC}"