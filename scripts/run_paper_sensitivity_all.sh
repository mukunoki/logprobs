#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
cd "$REPO_ROOT"

mkdir -p "${PAPER_DIR}/logs" "${RESULTS_DIR}"

NUM_TRIALS="${NUM_TRIALS:-20}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
PAPER_EASY4_MAX_TOKENS="${PAPER_EASY4_MAX_TOKENS:-4096}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/paper_sensitivity_${TS}}"
SUMMARY_JSON="${OUT_DIR}/sensitivity_summary.json"
mkdir -p "$OUT_DIR"

echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] num trials: $NUM_TRIALS"

run_eval() {
  local budget="$1"
  local temperature="$2"
  local output="$3"
  echo "[INFO] $(date '+%F %T') start sensitivity benchmark=paper_easy4 budget=${budget} temperature=${temperature}"
  python3 "${SCRIPTS_DIR}/threshold_refinement_eval.py" \
    --benchmark paper_easy4 \
    --num-trials "$NUM_TRIALS" \
    --budget "$budget" \
    --thresholds 0.80 0.85 0.90 \
    --temperature "$temperature" \
    --max-tokens "$PAPER_EASY4_MAX_TOKENS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --resume \
    --output "$output"
  echo "[INFO] $(date '+%F %T') done sensitivity benchmark=paper_easy4 budget=${budget} temperature=${temperature}"
}

run_eval 3 0.8 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t08.json"
run_eval 5 0.8 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b5_t08.json"
run_eval 7 0.8 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b7_t08.json"
run_eval 10 0.8 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b10_t08.json"

run_eval 3 0.2 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t02.json"
run_eval 3 0.5 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t05.json"
run_eval 3 1.0 "${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t10.json"

echo "[INFO] $(date '+%F %T') start sensitivity summary"
python3 "${SCRIPTS_DIR}/summarize_sensitivity_results.py" \
  --k-run "3=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t08.json" \
  --k-run "5=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b5_t08.json" \
  --k-run "7=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b7_t08.json" \
  --k-run "10=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b10_t08.json" \
  --temp-run "0.2=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t02.json" \
  --temp-run "0.5=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t05.json" \
  --temp-run "0.8=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t08.json" \
  --temp-run "1.0=${OUT_DIR}/threshold_refinement_eval_paper_easy4_b3_t10.json" \
  --output "$SUMMARY_JSON"
echo "[INFO] $(date '+%F %T') done sensitivity summary"

echo "[INFO] $(date '+%F %T') start sensitivity plotting"
CBBA_SENSITIVITY_JSON="$SUMMARY_JSON" python3 "${PAPER_DIR}/create_sensitivity_figure.py"
echo "[INFO] $(date '+%F %T') done sensitivity plotting"

echo "[INFO] completed sensitivity experiments"
echo "[INFO] summary json: $SUMMARY_JSON"
