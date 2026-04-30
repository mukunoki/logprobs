#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR"

NUM_TRIALS="${NUM_TRIALS:-10}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/hard4_prompt_minimal_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
OUT_JSON="${OUT_DIR}/threshold_refinement_eval_paper_hard4_b${BUDGET}.json"
mkdir -p "$OUT_DIR"

export VLLM_USE_CHAT

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] benchmark: paper_hard4"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] budget: $BUDGET"
echo "[INFO] temperature: $TEMPERATURE"
echo "[INFO] max_tokens: $MAX_TOKENS"
echo "[INFO] workers: $WORKERS"
echo "[INFO] vllm_use_chat: $VLLM_USE_CHAT"
echo "[INFO] $(date '+%F %T') start hard4 prompt check"

python3 "${SCRIPTS_DIR}/threshold_refinement_eval_parallel.py" \
  --benchmark paper_hard4 \
  --num-trials "$NUM_TRIALS" \
  --budget "$BUDGET" \
  --thresholds 0.80 0.85 0.90 \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --workers "$WORKERS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --resume \
  --output "$OUT_JSON"

echo "[INFO] $(date '+%F %T') done hard4 prompt check"
echo "[INFO] output json: $OUT_JSON"
