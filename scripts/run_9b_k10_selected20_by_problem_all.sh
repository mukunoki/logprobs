#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_DIR"

NUM_TRIALS="${NUM_TRIALS:-100}"
BUDGET="${BUDGET:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
COMPACT_OUTPUT="${COMPACT_OUTPUT:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
MODEL_TAG="${MODEL_TAG:-qwen35_9b_awq4}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/selected20_9b_k10_by_problem_${MODEL_TAG}_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
BY_PROBLEM_DIR="${BY_PROBLEM_DIR:-${OUT_DIR}/by_problem}"
ANALYSIS_MD="${ANALYSIS_MD:-${OUT_DIR}/selected20_9b_k10_by_problem_method_analysis.md}"
ANALYSIS_JSON="${ANALYSIS_JSON:-${OUT_DIR}/selected20_9b_k10_by_problem_method_analysis.json}"

mkdir -p "$OUT_DIR" "$BY_PROBLEM_DIR"
export VLLM_USE_CHAT VLLM_MODEL_NAME

PROBLEMS=(
  banded_edit_distance_i32
  cholesky_spd_f64
  conv2d_3x3_multi_channel
  crop2d_strided_u8
  csr_spmv_axpy_dot
  floyd_warshall_blocked
  heap_sort_implementation
  lower_tri_solve_strided_f64
  matrix_transpose_cache
  pareval_convex_hull_perimeter_f64
  pareval_largest_component_i32
  pareval_sort_ignore_zero_i32
  quadratic_roots_stable_f64
  radix_sort_u32_pairs
  rmsnorm_mixed
  stencil2d_halo5
  stencil3d_mixed_7pt
  topk_ignore_nan_f32
  transpose_strided_f64
  utf8_validate_strict
)

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] by-problem dir: $BY_PROBLEM_DIR"
echo "[INFO] benchmark: paper_9b_k10_selected20"
echo "[INFO] model: $VLLM_MODEL_NAME"
echo "[INFO] num trials: $NUM_TRIALS"
echo "[INFO] budget: $BUDGET"
echo "[INFO] temperature: $TEMPERATURE"
echo "[INFO] max_tokens: $MAX_TOKENS"
echo "[INFO] workers: $WORKERS"
echo "[INFO] checkpoint_every: $CHECKPOINT_EVERY"
echo "[INFO] compact_output: $COMPACT_OUTPUT"
echo "[INFO] vllm_use_chat: $VLLM_USE_CHAT"

compact_args=()
if [[ "$COMPACT_OUTPUT" == "1" ]]; then
  compact_args+=(--compact-output)
fi

for problem in "${PROBLEMS[@]}"; do
  output="${BY_PROBLEM_DIR}/${problem}.json"
  echo "[INFO] starting problem: ${problem}"
  python3 "${SCRIPTS_DIR}/same_prompt_eval_parallel.py" \
    --benchmark paper_9b_k10_selected20 \
    --problem-name "$problem" \
    --num-trials "$NUM_TRIALS" \
    --budget "$BUDGET" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --workers "$WORKERS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --resume \
    --output "$output" \
    "${compact_args[@]}"
  echo "[INFO] completed problem: ${problem}"
done

python3 "${SCRIPTS_DIR}/analyze_sameprompt_methods.py" \
  --input-dir "$BY_PROBLEM_DIR" \
  --output-md "$ANALYSIS_MD" \
  --output-json "$ANALYSIS_JSON"

echo "[INFO] completed selected20 9B k10 by-problem same-prompt evaluation"
echo "[INFO] by-problem dir: $BY_PROBLEM_DIR"
echo "[INFO] analysis: $ANALYSIS_MD"
