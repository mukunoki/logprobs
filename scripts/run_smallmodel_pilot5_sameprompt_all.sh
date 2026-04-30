#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR"

NUM_TRIALS="${NUM_TRIALS:-20}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
MODEL_TAG="${MODEL_TAG:-qwen35_9b_awq4}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/smallmodel_pilot5_${MODEL_TAG}_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
OUTPUT="${OUTPUT:-${OUT_DIR}/same_prompt_eval_paper_smallmodel_pilot5_b${BUDGET}.json}"
ANALYSIS_MD="${ANALYSIS_MD:-${OUT_DIR}/smallmodel_pilot5_method_analysis.md}"
ANALYSIS_JSON="${ANALYSIS_JSON:-${OUT_DIR}/smallmodel_pilot5_method_analysis.json}"

mkdir -p "$OUT_DIR"
export VLLM_USE_CHAT VLLM_MODEL_NAME

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] benchmark: paper_smallmodel_pilot5"
echo "[INFO] model: $VLLM_MODEL_NAME"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] budget: $BUDGET"
echo "[INFO] temperature: $TEMPERATURE"
echo "[INFO] max_tokens: $MAX_TOKENS"
echo "[INFO] workers: $WORKERS"
echo "[INFO] vllm_use_chat: $VLLM_USE_CHAT"

python3 "${SCRIPTS_DIR}/same_prompt_eval_parallel.py" \
  --benchmark paper_smallmodel_pilot5 \
  --num-trials "$NUM_TRIALS" \
  --budget "$BUDGET" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --workers "$WORKERS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --resume \
  --output "$OUTPUT"

python3 "${SCRIPTS_DIR}/analyze_sameprompt_methods.py" \
  --input "$OUTPUT" \
  --output-md "$ANALYSIS_MD" \
  --output-json "$ANALYSIS_JSON"

echo "[INFO] completed smallmodel pilot5 same-prompt evaluation"
echo "[INFO] output: $OUTPUT"
echo "[INFO] analysis: $ANALYSIS_MD"
