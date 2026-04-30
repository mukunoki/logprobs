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

NUM_TRIALS="${NUM_TRIALS:-100}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/paper12_sameprompt_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
HYBRID_OUT_DIR="${OUT_DIR}/hybrid"
FIGURE_OUT_DIR="${FIGURE_OUT_DIR:-${OUT_DIR}/figures}"
mkdir -p "$OUT_DIR" "$HYBRID_OUT_DIR"

export VLLM_USE_CHAT

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] generation protocol: same_prompt_n"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] budget: $BUDGET"
echo "[INFO] temperature: $TEMPERATURE"
echo "[INFO] max_tokens: $MAX_TOKENS"
echo "[INFO] workers: $WORKERS"
echo "[INFO] vllm_use_chat: $VLLM_USE_CHAT"

run_same_prompt_eval() {
  local benchmark="$1"
  local output="$2"
  echo "[INFO] $(date '+%F %T') start same-prompt eval benchmark=${benchmark}"
  python3 "${SCRIPTS_DIR}/same_prompt_eval_parallel.py" \
    --benchmark "$benchmark" \
    --num-trials "$NUM_TRIALS" \
    --budget "$BUDGET" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --resume \
    --output "$output"
  echo "[INFO] $(date '+%F %T') done same-prompt eval benchmark=${benchmark}"
}

run_same_prompt_eval paper_easy4 "${OUT_DIR}/same_prompt_eval_paper_easy4_b${BUDGET}.json"
run_same_prompt_eval paper_medium4 "${OUT_DIR}/same_prompt_eval_paper_medium4_b${BUDGET}.json"
run_same_prompt_eval paper_hard4 "${OUT_DIR}/same_prompt_eval_paper_hard4_b${BUDGET}.json"

echo "[INFO] $(date '+%F %T') start evaluate_hybrid_cbba"
python3 "${SCRIPTS_DIR}/evaluate_hybrid_cbba.py" \
  --paper-easy4 "${OUT_DIR}/same_prompt_eval_paper_easy4_b${BUDGET}.json" \
  --paper-medium4 "${OUT_DIR}/same_prompt_eval_paper_medium4_b${BUDGET}.json" \
  --paper-hard4 "${OUT_DIR}/same_prompt_eval_paper_hard4_b${BUDGET}.json" \
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

echo "[INFO] completed same-prompt paper12 experiments"
echo "[INFO] hybrid dir: $HYBRID_OUT_DIR"
echo "[INFO] figure dir: $FIGURE_OUT_DIR"
