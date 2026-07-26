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
# Priority: .venv/bin/python3 > python3 > python
# Works on CPU / GPU / TPU / Apple Silicon
detect_python() {
    # 1. Check venv first (from uv sync)
    if [ -f ".venv/bin/python3" ]; then
        echo ".venv/bin/python3"
        return 0
    fi
    if [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return 0
    fi
    
    # 2. Check system python3 (most common)
    if command -v python3 &> /dev/null; then
        # Verify it's Python 3.x
        PY_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo $PY_VER | cut -d. -f1)
        if [ "$MAJOR" -ge 3 ]; then
            echo "python3"
            return 0
        fi
    fi
    
    # 3. Fallback to python (some systems have python=python3)
    if command -v python &> /dev/null; then
        PY_VER=$(python --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo $PY_VER | cut -d. -f1)
        if [ "$MAJOR" -ge 3 ]; then
            echo "python"
            return 0
        fi
    fi
    
    # 4. No Python found
    echo "ERROR: No Python 3.x found. Install Python 3.11+"
    echo "  Linux:  apt install python3"
    echo "  macOS:  brew install python@3.11"
    echo "  Conda:  conda create -n trust python=3.11"
    exit 1
}

PYTHON=$(detect_python)
echo "  Using Python: ${PYTHON} ($($PYTHON --version 2>&1))"

# ─── Check PyTorch device (CPU/GPU/TPU/MPS) ─────────────────
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

# ─── Run ─────────────────────────────────────────────────────
echo "============================================================"
echo "  Trustworthiness Evaluation — Full Pipeline"
echo "  Date: $(date)"
echo "  Python: ${PYTHON}"
check_device
echo "============================================================"

# Step 1: Check Ollama
echo "[1/10] Checking prerequisites..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
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

# Step 2: Verify datasets
echo "[2/10] Verifying datasets..."
REQUIRED_FILES=(
    "data/final/safety.jsonl"
    "data/final/truthfulness.jsonl"
    "data/final/consistency.jsonl"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  $file"
    else
        echo "  ERROR: $file not found"
        exit 1
    fi
done

# Step 3: Clean previous results
echo "[3/10] Cleaning previous results..."
rm -rf "${RESULTS_DIR}/gemma3_4b" "${RESULTS_DIR}/llama3.1_8b"
rm -f "${RESULTS_DIR}/"*.json "${RESULTS_DIR}/"*.txt "${RESULTS_DIR}/"*.png
rm -f "${RESULTS_DIR}/raw_outputs/"*.jsonl

# Step 4: Run evaluation
echo "[4/10] Running evaluation (this may take 30-60 minutes)..."
START_TIME=$(date +%s)
$PYTHON run_evaluation.py --models "$MODELS" --output "$RESULTS_DIR"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Evaluation complete in ${DURATION}s"

# Step 5: Save pipeline summary
echo "[5/10] Saving pipeline summary..."
cat > "${RESULTS_DIR}/pipeline_summary.txt" << EOF
=================================================
  Pipeline Summary
=================================================
Date:        $(date)
Models:      ${MODELS}
Python:      ${PYTHON} ($($PYTHON --version 2>&1))
Duration:    ${DURATION}s
Device:      $(check_device | head -1)
Reproduce:   make run  OR  ./week2.sh
=================================================
EOF

# Step 6: Generate manual audit
echo "[6/10] Generating manual audit file..."
$PYTHON scripts/manual_audit_consistency.py

# Step 7: Generate analysis
echo "[7/10] Generating analysis plots..."
$PYTHON scripts/analysis.py

# Step 8: Offline rescoring verification
echo "[8/10] Running offline rescoring verification..."
$PYTHON scripts/score_saved_outputs.py \
    --input "${RESULTS_DIR}/raw_outputs/*.jsonl" \
    --output "${RESULTS_DIR}/rescored_verification.json" \
    --dimension all

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results saved to: ${RESULTS_DIR}/"
echo "  Reproduce:        make clean && make run"
echo "  Or:               ./week2.sh"
echo "============================================================"
