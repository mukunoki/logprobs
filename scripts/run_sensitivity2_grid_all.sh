#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR"

NUM_TRIALS="${NUM_TRIALS:-20}"
TEMPERATURES="${TEMPERATURES:-0.4 0.8 1.0}"
BUDGETS="${BUDGETS:-3 5 10}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/sensitivity2_grid_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"

mkdir -p "$OUT_DIR"
export VLLM_USE_CHAT

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] benchmark: paper_sensitivity2"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] temperatures: $TEMPERATURES"
echo "[INFO] budgets: $BUDGETS"
echo "[INFO] max_tokens: $MAX_TOKENS"
echo "[INFO] workers: $WORKERS"
echo "[INFO] vllm_use_chat: $VLLM_USE_CHAT"

for temperature in $TEMPERATURES; do
  for budget in $BUDGETS; do
    tag="t${temperature}_k${budget}"
    tag="${tag//./p}"
    output="${OUT_DIR}/same_prompt_eval_paper_sensitivity2_${tag}.json"
    analysis_md="${OUT_DIR}/sensitivity2_method_analysis_${tag}.md"
    analysis_json="${OUT_DIR}/sensitivity2_method_analysis_${tag}.json"
    echo "[INFO] $(date '+%F %T') start sensitivity combo temperature=${temperature} budget=${budget}"
    python3 "${SCRIPTS_DIR}/same_prompt_eval_parallel.py" \
      --benchmark paper_sensitivity2 \
      --num-trials "$NUM_TRIALS" \
      --budget "$budget" \
      --temperature "$temperature" \
      --max-tokens "$MAX_TOKENS" \
      --workers "$WORKERS" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --resume \
      --output "$output"

    python3 "${SCRIPTS_DIR}/analyze_sameprompt_methods.py" \
      --input "$output" \
      --output-md "$analysis_md" \
      --output-json "$analysis_json"
    echo "[INFO] $(date '+%F %T') done sensitivity combo temperature=${temperature} budget=${budget}"
  done
done

echo "[INFO] completed sensitivity2 grid"
echo "[INFO] out: $OUT_DIR"
