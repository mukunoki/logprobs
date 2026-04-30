#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_9b_k10_selected20_sameprompt_all.sh"
cd "$REPO_ROOT"

mkdir -p "${PAPER_DIR}/logs" "${RESULTS_DIR}"

NUM_TRIALS="${NUM_TRIALS:-50}"
BUDGET="${BUDGET:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
COMPACT_OUTPUT="${COMPACT_OUTPUT:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
MODEL_TAG="${MODEL_TAG:-qwen35_9b_awq4}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/selected20_9b_k10_${MODEL_TAG}_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
LOG_PATH="${LOG_PATH:-${PAPER_DIR}/logs/selected20_9b_k10_${MODEL_TAG}_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}.log}"
PID_PATH="${PAPER_DIR}/logs/selected20_9b_k10_latest.pid"
META_PATH="${PAPER_DIR}/logs/selected20_9b_k10_latest.meta"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A selected20 9B k10 experiment is already running with PID $old_pid" >&2
    exit 1
  fi
fi

export NUM_TRIALS BUDGET TEMPERATURE MAX_TOKENS WORKERS CHECKPOINT_EVERY COMPACT_OUTPUT VLLM_USE_CHAT VLLM_MODEL_NAME MODEL_TAG OUT_DIR LOG_PATH RUN_SCRIPT PYTHONUNBUFFERED=1

pid="$(
python3 - <<'PY'
import os
import subprocess

root = os.getcwd()
with open(os.environ["LOG_PATH"], "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["bash", os.environ["RUN_SCRIPT"]],
        cwd=root,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(proc.pid)
PY
)"

echo "$pid" > "$PID_PATH"
cat > "$META_PATH" <<EOF
pid=$pid
started_at=$(date '+%F %T')
log=$LOG_PATH
out_dir=$OUT_DIR
benchmark=paper_9b_k10_selected20
generation_protocol=same_prompt_n
model=$VLLM_MODEL_NAME
num_trials=$NUM_TRIALS
budget=$BUDGET
temperature=$TEMPERATURE
max_tokens=$MAX_TOKENS
workers=$WORKERS
checkpoint_every=$CHECKPOINT_EVERY
compact_output=$COMPACT_OUTPUT
vllm_use_chat=$VLLM_USE_CHAT
EOF

echo "Started selected20 9B k10 same-prompt background run"
echo "pid: $pid"
echo "log: $LOG_PATH"
echo "out_dir: $OUT_DIR"
