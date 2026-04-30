#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
PAPER_FIGURE_SCRIPT="${PAPER_DIR}/create_clean_figures_12problems.py"
cd "$REPO_ROOT"

mkdir -p "${PAPER_DIR}/logs" "${RESULTS_DIR}"

NUM_TRIALS="${NUM_TRIALS:-5}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
PAPER_EASY4_MAX_TOKENS="${PAPER_EASY4_MAX_TOKENS:-4096}"
PAPER_MEDIUM4_MAX_TOKENS="${PAPER_MEDIUM4_MAX_TOKENS:-4096}"
PAPER_HARD4_MAX_TOKENS="${PAPER_HARD4_MAX_TOKENS:-4096}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/paper12_k5_${TS}}"
HYBRID_OUT_DIR="${OUT_DIR}/hybrid"
FIGURE_OUT_DIR="${FIGURE_OUT_DIR:-${OUT_DIR}/figures}"
mkdir -p "$OUT_DIR" "$HYBRID_OUT_DIR"

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] budget: $BUDGET"
echo "[INFO] temperature: $TEMPERATURE"

run_threshold_eval() {
  local benchmark="$1"
  local max_tokens="$2"
  local output="$3"
  echo "[INFO] $(date '+%F %T') start threshold_refinement_eval benchmark=${benchmark} budget=${BUDGET} max_tokens=${max_tokens}"
  python3 "${SCRIPTS_DIR}/threshold_refinement_eval.py" \
    --benchmark "$benchmark" \
    --num-trials "$NUM_TRIALS" \
    --budget "$BUDGET" \
    --thresholds 0.80 0.85 0.90 \
    --temperature "$TEMPERATURE" \
    --max-tokens "$max_tokens" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --resume \
    --output "$output"
  echo "[INFO] $(date '+%F %T') done threshold_refinement_eval benchmark=${benchmark}"
}

run_threshold_eval paper_easy4 "$PAPER_EASY4_MAX_TOKENS" "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b${BUDGET}.json"
run_threshold_eval paper_medium4 "$PAPER_MEDIUM4_MAX_TOKENS" "${OUT_DIR}/threshold_refinement_eval_paper_medium4_b${BUDGET}.json"
run_threshold_eval paper_hard4 "$PAPER_HARD4_MAX_TOKENS" "${OUT_DIR}/threshold_refinement_eval_paper_hard4_b${BUDGET}.json"

echo "[INFO] $(date '+%F %T') start merge paper groups"
python3 "${SCRIPTS_DIR}/merge_threshold_eval_runs.py" \
  --benchmark-name paper12 \
  --output "${OUT_DIR}/threshold_refinement_eval_paper12_b${BUDGET}.json" \
  "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b${BUDGET}.json" \
  "${OUT_DIR}/threshold_refinement_eval_paper_medium4_b${BUDGET}.json" \
  "${OUT_DIR}/threshold_refinement_eval_paper_hard4_b${BUDGET}.json"
echo "[INFO] $(date '+%F %T') done merge paper groups"

echo "[INFO] $(date '+%F %T') start evaluate_hybrid_cbba"
python3 "${SCRIPTS_DIR}/evaluate_hybrid_cbba.py" \
  --paper-easy4 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b${BUDGET}.json" \
  --paper-medium4 "${OUT_DIR}/threshold_refinement_eval_paper_medium4_b${BUDGET}.json" \
  --paper-hard4 "${OUT_DIR}/threshold_refinement_eval_paper_hard4_b${BUDGET}.json" \
  --budget-paper-easy4 "$BUDGET" \
  --budget-paper-medium4 "$BUDGET" \
  --budget-paper-hard4 "$BUDGET" \
  --output-dir "$HYBRID_OUT_DIR"
echo "[INFO] $(date '+%F %T') done evaluate_hybrid_cbba"

echo "[INFO] $(date '+%F %T') start paper summary"
python3 "${SCRIPTS_DIR}/summarize_paper_results.py" \
  --hybrid-dir "$HYBRID_OUT_DIR" \
  --output-dir "$OUT_DIR"
echo "[INFO] $(date '+%F %T') done paper summary"

echo "[INFO] $(date '+%F %T') start plotting"
CBBA_HYBRID_DIR="$HYBRID_OUT_DIR" CBBA_FIGURE_OUT_DIR="$FIGURE_OUT_DIR" \
  python3 "$PAPER_FIGURE_SCRIPT"
echo "[INFO] $(date '+%F %T') done plotting"

echo "[INFO] completed unified k=5 experiments"
echo "[INFO] hybrid dir: $HYBRID_OUT_DIR"
echo "[INFO] figure dir: $FIGURE_OUT_DIR"
