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

# ─── Human-in-the-loop gateway ───────────────────────────────
# Pauses the pipeline until the target annotation file has been filled by a
# human. Runs only when stdin is a TTY (interactive shell); set RUN_INTERACTIVE=0
# to force-skip gates in automated / CI runs.
is_interactive() {
    [ -t 0 ] && [ "${RUN_INTERACTIVE:-1}" = "1" ]
}

# Args: $1 = jsonl file, $2 = label field. Prints "filled/total".
count_labelled() {
    "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
filepath, field = sys.argv[1], sys.argv[2]
total = filled = 0
try:
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if rec.get(field) not in (None, ""):
                filled += 1
    print(f"{filled}/{total}")
except FileNotFoundError:
    print("0/0")
PY
}

# Args: $1 = jsonl file, $2 = label field, $3 = human instructions.
wait_for_annotation() {
    local file="$1" field="$2" prompt="$3"
    if ! is_interactive; then
        echo "  [non-interactive] skipping wait for: ${file}"
        return 0
    fi
    if [ ! -f "$file" ]; then
        echo "  [gate] file not found: ${file}"
        return 0
    fi
    echo ""
    echo "  =================================================="
    echo "  ⏸  HUMAN-IN-THE-LOOP STEP"
    echo "  File: ${file}"
    echo "  --------------------------------------------------"
    echo "  $prompt"
    echo ""
    while true; do
        local counts filled total
        counts=$(count_labelled "$file" "$field")
        filled=${counts%/*}
        total=${counts#*/}
        echo "  Progress: ${filled} / ${total} labelled"
        if [ -n "$total" ] && [ "$total" -gt 0 ] && [ "$filled" -ge "$total" ]; then
            echo "  ✓ All records labelled. Continuing."
            break
        fi
        if [ "$total" = "0" ]; then
            echo "  ⚠  No records found in ${file}."
            break
        fi
        echo ""
        read -r -p "  When done annotating, press [Enter] to continue (type 'skip' to bypass): " ans
        if [ "$ans" = "skip" ]; then
            echo "  ⚠  Gate skipped — reports may be incomplete."
            break
        fi
    done
    echo ""
}

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

echo "[1/14] Checking prerequisites..."
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

echo "[2/14] Verifying datasets..."
for file in "data/final/safety.jsonl" "data/final/truthfulness.jsonl" "data/final/consistency.jsonl"; do
    if [ -f "$file" ]; then echo "  $file"; else echo "  ERROR: $file not found"; exit 1; fi
done

echo "[3/14] Cleaning previous results..."
rm -rf "${RESULTS_DIR}/gemma3_4b" "${RESULTS_DIR}/llama3.1_8b"
rm -f "${RESULTS_DIR}/"*.json "${RESULTS_DIR}/"*.txt "${RESULTS_DIR}/"*.png
rm -f "${RESULTS_DIR}/raw_outputs/"*.jsonl
rm -f "${RESULTS_DIR}/audit/agreement_report.json"

echo "[4/14] Running evaluation (may take 30-60 minutes)..."
START_TIME=$(date +%s)
$PYTHON run_evaluation.py --models "$MODELS" --output "$RESULTS_DIR"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Evaluation complete in ${DURATION}s"

echo "[5/14] Regenerating audit dataset from fresh raw outputs..."
# Rebuild results/audit/all_audit.jsonl FROM the current run so the dashboard
# and blinded annotation never show stale / hand-labelled data from an older
# evaluation. Human labels start NULL (ready to annotate).
$PYTHON scripts/generate_audit_samples.py \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output "${RESULTS_DIR}/audit/all_audit.jsonl" \
    --n-safety 10 --n-truthfulness 10 --n-consistency 10 --seed 42
# Rebuild the blinded calibration / held-out splits from the same fresh data.
$PYTHON scripts/generate_blinded_annotation.py \
    --audit "${RESULTS_DIR}/audit/all_audit.jsonl" \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output "${RESULTS_DIR}/audit/blinded" 2>/dev/null \
    || echo "  (skip blinded split generation — optional, manual audit still works)"

echo "[6/14] Saving pipeline summary..."
cat > "${RESULTS_DIR}/pipeline_summary.txt" << EOF
Pipeline Summary
Date:     $(date '+%a, %d %b %Y %H:%M:%S')
Models:   ${MODELS}
Python:   ${PYTHON} ($($PYTHON --version 2>&1))
Duration: ${DURATION}s
Reproduce: ./demo.sh
EOF

echo "[7/14] Generating analysis plots..."
$PYTHON scripts/analysis.py

echo "[8/14] Running offline rescoring verification..."
$PYTHON scripts/score_saved_outputs.py     --input "${RESULTS_DIR}/raw_outputs/*.jsonl"     --output "${RESULTS_DIR}/rescored_verification.json"     --dimension all

echo "[9/14] Generating manual audit file..."
$PYTHON scripts/manual_audit_consistency.py
# HUMAN GATE: pause until a human labels the consistency pairs in the dashboard
# file. The paradigm report (step 11) reads all_audit.jsonl, whose fresh human
# labels should be filled first so κ figures are current.
wait_for_annotation \
    "${RESULTS_DIR}/manual_audit_consistency.jsonl" \
    "human_label" \
    "Fill every 'human_label' in results/manual_audit_consistency.jsonl
     (consistent / inconsistent) for the manual consistency audit.
     Tip: you can also do this later from the dashboard's Manual Audit tab."

echo "[10/14] Filling audit human labels (for Research Question tab)..."
echo "  Fill results/audit/all_audit.jsonl 'human_label' fields (correct /
  incorrect / consistent / inconsistent) to power agreement/κ metrics."
wait_for_annotation \
    "${RESULTS_DIR}/audit/all_audit.jsonl" \
    "human_label" \
    "Fill every 'human_label' in results/audit/all_audit.jsonl
     (correct / incorrect for safety & truthfulness;
      consistent / inconsistent for consistency)."

echo "[11/14] Generating paradigm report..."
# Generate agreement_report.json + validation_report.json from the now-filled
# audit labels so the dashboard Research Question / Human Annotation tabs show
# current κ figures for this run.
$PYTHON scripts/paradigm_report.py

echo "[12/14] Preparing blinded annotation templates..."
# Emit one annotation template per annotator (blinded: no auto_label/similarity).
# A human annotator then fills in human_label / confidence / notes for each record.
if [ -f "${RESULTS_DIR}/audit/blinded/blinded_annotation_calibration.jsonl" ]; then
  $PYTHON scripts/run_blinded_annotation.py prepare \
      --input "${RESULTS_DIR}/audit/blinded/blinded_annotation_calibration.jsonl" \
      --output "${RESULTS_DIR}/blinded_work" \
      --annotators ann1 ann2
  echo "  → Edit results/blinded_work/ann1.jsonl and ann2.jsonl, then run 'make blinded-report'."
  # HUMAN GATE: wait for ≥2 independent annotators to fill their templates.
  for ann in ann1 ann2; do
    wait_for_annotation \
        "${RESULTS_DIR}/blinded_work/${ann}.jsonl" \
        "human_label" \
        "Fill every 'human_label' in results/blinded_work/${ann}.jsonl
         (correct / incorrect / consistent / inconsistent). Keep annotators
         BLINDED — do not compare with each other or the auto labels."
  done
else
  echo "  Skipping blinded annotation (${RESULTS_DIR}/audit/blinded/blinded_annotation_calibration.jsonl not found)"
fi

echo "[13/14] Building blinded re-annotation report (if templates are filled)..."
# Only build the inter-annotator report if an annotation file exists AND has been
# filled by a human (non-empty human_label); otherwise the pipeline must NOT fail.
if [ -f "${RESULTS_DIR}/blinded_work/ann1.jsonl" ] && [ -f "${RESULTS_DIR}/blinded_work/ann2.jsonl" ]; then
  if grep -q '"human_label": "[^"]' "${RESULTS_DIR}/blinded_work/ann1.jsonl"; then
    echo "  Inter-annotator agreement (DIMENSION=all):"
    $PYTHON scripts/run_blinded_annotation.py report \
        --annotations "${RESULTS_DIR}/blinded_work/ann1.jsonl" "${RESULTS_DIR}/blinded_work/ann2.jsonl" \
        --dimension all \
        --audit "${RESULTS_DIR}/audit/all_audit.jsonl" \
        --output "${RESULTS_DIR}/audit/inter_annotator_report.json" || echo "  (report failed — continuing)"
  else
    echo "  Skipping report (annotation templates not filled yet by humans)"
  fi
else
  echo "  Skipping report (annotation templates not present)"
fi

echo "[14/14] Starting Streamlit dashboard..."
echo "  Open: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""
$PYTHON -m streamlit run app/dashboard.py

echo ""
echo "============================================================"
echo "  Pipeline finished!"
echo "============================================================"


