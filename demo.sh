#!/usr/bin/env bash
# ============================================================
# Trustworthiness Evaluation — Full Pipeline
# Auto-detects Python executable (python3 or python)
# Works on CPU, GPU, TPU, MPS (Apple Silicon)
# ============================================================
set -e

MODELS="gemma3:4b,llama3.1:8b"
RESULTS_DIR="results"

# ─── Python Detection ───────────────────────────────────────
detect_python() {
    if [ -f ".venv/bin/python3" ]; then
        echo ".venv/bin/python3"
        return 0
    fi
    if [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return 0
    fi
    if command -v python3 &> /dev/null; then
        PY_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo $PY_VER | cut -d. -f1)
        if [ "$MAJOR" -ge 3 ]; then
            echo "python3"
            return 0
        fi
    fi
    if command -v python &> /dev/null; then
        PY_VER=$(python --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo $PY_VER | cut -d. -f1)
        if [ "$MAJOR" -ge 3 ]; then
            echo "python"
            return 0
        fi
    fi
    echo "ERROR: No Python 3.x found. Install Python 3.11+"
    exit 1
}

PYTHON=$(detect_python)
echo "  Using Python: ${PYTHON} ($($PYTHON --version 2>&1))"

# ─── Check device ───────────────────────────────────────────
check_device() {
    DEVICE=$($PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('Apple Silicon (MPS)')
else:
    print('CPU')
" 2>/dev/null || echo "CPU (no torch)")
    echo "  Device: ${DEVICE}"
}

# ─── Run Pipeline ───────────────────────────────────────────
echo "============================================================"
echo "  Trustworthiness Evaluation — Full Pipeline"
echo "  Date: $(date '+%a, %d %b %Y %H:%M:%S')"
echo "  Python: ${PYTHON}"
check_device
echo "============================================================"

echo "[1/10] Checking prerequisites..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed"
    exit 1
fi
echo "  Ollama: $(ollama --version 2>&1 || echo 'installed')"

for model in $(echo $MODELS | tr ',' ' '); do
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo "  Model '$model' available"
    else
        echo "  Pulling model '$model'..."
        ollama pull "$model"
    fi
done

echo "[2/10] Verifying datasets..."
for file in "data/final/safety.jsonl" "data/final/truthfulness.jsonl" "data/final/consistency.jsonl"; do
    if [ -f "$file" ]; then echo "  $file"; else echo "  ERROR: $file not found"; exit 1; fi
done

echo "[3/10] Cleaning previous results..."
rm -rf "${RESULTS_DIR}/gemma3_4b" "${RESULTS_DIR}/llama3.1_8b"
rm -f "${RESULTS_DIR}/"*.json "${RESULTS_DIR}/"*.txt "${RESULTS_DIR}/"*.png
rm -f "${RESULTS_DIR}/raw_outputs/"*.jsonl
rm -f "${RESULTS_DIR}/audit/agreement_report.json"

echo "[4/10] Running evaluation (may take 30-60 minutes)..."
START_TIME=$(date +%s)
$PYTHON run_evaluation.py --models "$MODELS" --output "$RESULTS_DIR"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Evaluation complete in ${DURATION}s"

echo "[5/10] Saving pipeline summary..."
cat > "${RESULTS_DIR}/pipeline_summary.txt" << EOF
Pipeline Summary
Date:     $(date '+%a, %d %b %Y %H:%M:%S')
Models:   ${MODELS}
Python:   ${PYTHON} ($($PYTHON --version 2>&1))
Duration: ${DURATION}s
Reproduce: ./demo.sh
EOF

echo "[6/10] Generating manual audit file..."
$PYTHON scripts/manual_audit_consistency.py

echo "[7/10] Generating analysis plots..."
$PYTHON scripts/analysis.py

echo "[8/10] Running offline rescoring verification..."
$PYTHON scripts/score_saved_outputs.py     --input "${RESULTS_DIR}/raw_outputs/*.jsonl"     --output "${RESULTS_DIR}/rescored_verification.json"     --dimension all

echo "[9/10] Generating paradigm report..."
$PYTHON scripts/paradigm_report.py

echo "[10/10] Starting Streamlit dashboard..."
echo "  Open: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""
$PYTHON -m streamlit run app/dashboard.py

echo ""
echo "============================================================"
echo "  Pipeline finished!"
echo "============================================================"

