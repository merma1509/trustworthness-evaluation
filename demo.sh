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
            echo "   No records found in ${file}."
            break
        fi
        echo ""
        read -r -p "  When done annotating, press [Enter] to continue (type 'skip' to bypass): " ans
        if [ "$ans" = "skip" ]; then
            echo "   Gate skipped — reports may be incomplete."
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

echo "[1/12] Checking prerequisites..."
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

echo "[2/12] Verifying datasets..."
for file in "data/final/safety.jsonl" "data/final/truthfulness.jsonl" "data/final/consistency.jsonl"; do
    if [ -f "$file" ]; then echo "  $file"; else echo "  ERROR: $file not found"; exit 1; fi
done

echo "[3/12] Cleaning previous results..."
rm -rf "${RESULTS_DIR}/gemma3_4b" "${RESULTS_DIR}/llama3.1_8b"
rm -f "${RESULTS_DIR}/"*.json "${RESULTS_DIR}/"*.txt "${RESULTS_DIR}/"*.png
rm -f "${RESULTS_DIR}/raw_outputs/"*.jsonl
rm -f "${RESULTS_DIR}/audit/agreement_report.json"

echo "[4/12] Running evaluation (may take 30-60 minutes)..."
START_TIME=$(date +%s)
$PYTHON run_evaluation.py --models "$MODELS" --output "$RESULTS_DIR"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Evaluation complete in ${DURATION}s"

echo "[5a/12] Regenerating audit dataset from fresh raw outputs..."
# Rebuild results/audit/all_audit.jsonl FROM the current run so the dashboard
# and blinded annotation never show stale / hand-labelled data from an older
# evaluation. Human labels start NULL (ready to annotate).
$PYTHON scripts/generate_audit_samples.py \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output "${RESULTS_DIR}/audit/all_audit.jsonl" \
    --n-safety 10 --n-truthfulness 10 --n-consistency 10 --seed 42
# Regenerate the FULL-dataset blinded experiment audit (experiment/all_audit_full.jsonl)
# and rebuild the anonymised calibration/held-out splits (experiment/blinded/) from the
# same fresh raw outputs. This is the single blinded-annotation flow the pipeline uses.
$PYTHON scripts/generate_audit_samples.py \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output experiment/all_audit_full.jsonl \
    --n-safety 100 --n-truthfulness 100 --n-consistency 100 --seed 42
$PYTHON scripts/generate_blinded_annotation.py \
    --audit experiment/all_audit_full.jsonl \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output experiment/blinded 2>/dev/null \
    || echo "  (skip blinded split generation — optional, manual audit still works)"

echo "[5b/12] Regenerating human-timing study from fresh records..."
# Results cost/budget analysis reads results/human_timing_measurement.json as the
# SINGLE SOURCE OF TRUTH for per-label human time. Step 3 wiped results/*.json,
# so regenerate it here from the just-rebuilt audit + raw outputs — otherwise RQ4
# (cost) silently falls back to the 30s placeholder.
$PYTHON scripts/generate_human_timing_from_records.py \
    --audit "${RESULTS_DIR}/audit/all_audit.jsonl" \
    --raw "${RESULTS_DIR}/raw_outputs" \
    --output "${RESULTS_DIR}/human_timing_measurement.json" \
    --seed 42 2>/dev/null \
    || echo "  (human-timing generation skipped — RQ4 cost will use the 30s placeholder)"
echo "  → human_timing_measurement.json regenerated from actual records (Task 1.5)."

echo "[6/12] Saving pipeline summary..."
cat > "${RESULTS_DIR}/pipeline_summary.txt" << EOF
Pipeline Summary
Date:     $(date '+%a, %d %b %Y %H:%M:%S')
Models:   ${MODELS}
Python:   ${PYTHON} ($($PYTHON --version 2>&1))
Duration: ${DURATION}s
Reproduce: ./demo.sh
EOF

echo "[7/12] Generating analysis plots..."
$PYTHON scripts/analysis.py

echo "[8/12] Running offline rescoring verification..."
$PYTHON scripts/score_saved_outputs.py     --input "${RESULTS_DIR}/raw_outputs/*.jsonl"     --output "${RESULTS_DIR}/rescored_verification.json"     --dimension all

echo "[9/12] Generating manual audit file..."
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

echo "[10/12] Filling audit human labels (for Research Question tab)..."
echo "  Fill results/audit/all_audit.jsonl 'human_label' fields (correct /
  incorrect / consistent / inconsistent) to power agreement/κ metrics."
wait_for_annotation \
    "${RESULTS_DIR}/audit/all_audit.jsonl" \
    "human_label" \
    "Fill every 'human_label' in results/audit/all_audit.jsonl
     (correct / incorrect for safety & truthfulness;
      consistent / inconsistent for consistency)."

echo "[11a/12] Generating paradigm report..."
# Generate agreement_report.json + validation_report.json from the now-filled
# audit labels so the dashboard Research Question / Human Annotation tabs show
# current κ figures for this run.
$PYTHON scripts/paradigm_report.py --with-cost

echo "[11b/12] Generating figures & budget (if reports exist)..."
# The budget / figure scripts read validation_report.json + agreement_report.json
# produced in step 11, so they run *after* the human labels are filled. All are
# optional: if a source report is absent the scripts degrade gracefully.
if [ -f "${RESULTS_DIR}/validation_report.json" ]; then
  $PYTHON scripts/budget_reliability_curve.py \
      --report "${RESULTS_DIR}/validation_report.json" \
      --output "${RESULTS_DIR}/budget_reliability_curve.png" \
      || echo "  (skip budget-vs-reliability figure)"
  $PYTHON scripts/budget_optimizer.py \
      --report "${RESULTS_DIR}/validation_report.json" \
      --output "${RESULTS_DIR}/budget_plan.json" \
      || echo "  (skip budget plan)"
fi
if [ -f "${RESULTS_DIR}/audit/agreement_report.json" ]; then
  $PYTHON scripts/error_heatmap.py \
      --agreement "${RESULTS_DIR}/audit/agreement_report.json" \
      --validation "${RESULTS_DIR}/validation_report.json" \
      --output "${RESULTS_DIR}/error_heatmap.png" \
      || echo "  (skip error heatmap)"
  $PYTHON scripts/pipeline_diagram.py \
      --output "${RESULTS_DIR}/pipeline_loop.png" \
      || echo "  (skip pipeline-loop diagram)"
fi
echo "  → Artifacts: budget_plan.json, budget_reliability_curve.png, error_heatmap.png, pipeline_loop.png"

echo "[11c/12] Experiment flow status..."
# The blinded full-dataset experiment (≥2 independent, blinded annotators) is the
# single blinded-annotation flow. Step 5a built experiment/all_audit_full.jsonl +
# experiment/blinded; this block only reports the next human step so a fresh
# `make run` never blocks on annotators who aren't present yet.
if [ -f "experiment/all_audit_full.jsonl" ] && [ -d "experiment/blinded" ]; then
  echo "  → Experiment audit + anonymised splits are ready (experiment/all_audit_full.jsonl)."
  FILLED=0
  TOTAL=0
  for f in experiment/held_out_work/*.jsonl; do
    [ -f "$f" ] || continue
    c=$(count_labelled "$f" "human_label")
    filled=${c%/*}; total=${c#*/}
    TOTAL=$((TOTAL + total)); FILLED=$((FILLED + filled))
  done
  if [ "$TOTAL" -gt 0 ] && [ "$FILLED" -ge "$TOTAL" ]; then
    echo "  → Held-out templates fully filled (${FILLED}/${TOTAL})."
    echo "  → Produce the blinded report with:"
    echo "      make experiment-heldout-report \\"
    echo "           ANNOTATIONS=\"experiment/held_out_work/ann1.jsonl experiment/held_out_work/ann2.jsonl\""
    echo "      make experiment-budget"
  elif [ "$TOTAL" -gt 0 ]; then
    echo "  → Held-out templates exist but not yet fully filled (${FILLED}/${TOTAL})."
    echo "  → ≥2 independent annotators must fill experiment/held_out_work/*.jsonl, then run"
    echo "      make experiment-heldout-report   (and optionally make experiment-budget)"
  else
    echo "  → No held-out templates yet. Prepare them with:"
    echo "      make experiment-heldout-prepare ANNOTATORS=\"ann1 ann2\""
  fi
  echo "  → Verify anonymity first with: make experiment-blinded-verify"
else
  echo "  → Experiment flow not started. To enable full-dataset blinded validation:"
  echo "      make experiment-audit"
  echo "      make experiment-prepare"
  echo "      make experiment-blinded-verify"
fi

echo "[12/12] Starting Streamlit dashboard..."
echo "  Open: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""
$PYTHON -m streamlit run app/dashboard.py

echo ""
echo "============================================================"
echo "  Pipeline finished!"
echo "============================================================"


