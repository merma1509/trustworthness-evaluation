#!/usr/bin/env bash
# ============================================================
# Trustworthiness Evaluation — Week 1 Pipeline
# Runs the full evaluation pipeline and saves timestamped results
# ============================================================
set -e  # Exit on any error

# --- Configuration ---
MODELS="gemma3:4b,llama3.1:8b"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${TIMESTAMP}"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Trustworthiness Evaluation — Week 1 Pipeline${NC}"
echo -e "${BLUE}  Date: $(date)${NC}"
echo -e "${BLUE}  Models: ${MODELS}${NC}"
echo -e "${BLUE}  Output: ${RESULTS_DIR}/${NC}"
echo -e "${BLUE}============================================================${NC}"

# --- Step 1: Check prerequisites ---
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}  Python3 found${NC}"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}  Ollama found${NC}"

# Check models
for model in $(echo $MODELS | tr ',' ' '); do
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo -e "${GREEN}  Model '$model' available${NC}"
    else
        echo -e "${YELLOW} Model '$model' not found. Pulling...${NC}"
        ollama pull "$model"
    fi
done

# --- Step 2: Verify datasets ---
echo -e "\n${YELLOW}[2/6] Verifying datasets...${NC}"

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
echo -e "\n${YELLOW}[3/6] Validating raw seeds...${NC}"
python3 scripts/validate_raw_seeds.py 2>&1 | tail -5
echo -e "${GREEN}  Seed validation complete${NC}"

# --- Step 4: Run evaluation pipeline ---
echo -e "\n${YELLOW}[4/6] Running evaluation pipeline...${NC}"
echo -e "  Models: ${MODELS}"
echo -e "  Output: ${RESULTS_DIR}/"

START_TIME=$(date +%s)

python3 run_evaluation.py \
    --models "$MODELS" \
    --output "$RESULTS_DIR"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}  Evaluation complete in ${DURATION}s${NC}"

# --- Step 5: Save a run summary ---
echo -e "\n${YELLOW}[5/6] Saving run summary...${NC}"

# Create summary
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

# --- Step 6: Print results ---
echo -e "\n${YELLOW}[6/6] Results${NC}"
echo -e "${BLUE}============================================================${NC}"

RESULTS_FILE="${RESULTS_DIR}/results_summary.json"
if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}Results saved to: ${RESULTS_DIR}/${NC}"
    echo ""
    
    # Pretty-print the key results using Python
    python3 -c "
import json
with open('${RESULTS_FILE}') as f:
    data = json.load(f)

results = data.get('results', {})
print('  Model Scores:')
for model in sorted(results.keys()):
    r = results[model]
    ds = r.get('dimension_scores', {})
    trust = r.get('trustworthiness_score', 'N/A')
    s = ds.get('safety', {})
    t = ds.get('truthfulness', {})
    c = ds.get('consistency', {})
    print(f'    {model}:')
    print(f'      Safety:       {s.get(\"score\", 0):.4f}')
    print(f'      Truthfulness: {t.get(\"score\", 0):.4f}')
    print(f'      Consistency:  {c.get(\"score\", 0):.4f}')
    print(f'      TrustScore:   {trust:.4f}')
    print()
"
fi

# Check ranking stability
if [ -f "${RESULTS_DIR}/ranking_stability.json" ]; then
    python3 -c "
import json
with open('${RESULTS_DIR}/ranking_stability.json') as f:
    data = json.load(f)

print('  Ranking Stability:')
for config in data.get('configurations', []):
    ranking = config.get('ranking', {})
    scores = config.get('scores', {})
    rank_str = ' > '.join([f'{m} (rank {r})' for m, r in sorted(ranking.items(), key=lambda x: x[1])])
    print(f'    {config[\"config\"]}: {rank_str}')
"
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}  Pipeline complete!${NC}"
echo -e "${GREEN}  Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${BLUE}============================================================${NC}"

# Create a symlink to the latest results for easy access
ln -sfn "${RESULTS_DIR}" results_latest
echo -e "  Symlink created: results_latest -> ${RESULTS_DIR}"