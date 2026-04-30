#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
RUN_SCRIPT="${REPO_ROOT}/scripts/threshold_refinement_eval_parallel.py"
cd "$REPO_ROOT"

mkdir -p "${PAPER_DIR}/logs" "${RESULTS_DIR}"

NUM_TRIALS="${NUM_TRIALS:-100}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/competitive4_parallel_${TS}}"
OUTPUT="${OUTPUT:-${OUT_DIR}/threshold_refinement_eval_paper_competitive4_b${BUDGET}.json}"
LOG_PATH="${LOG_PATH:-${PAPER_DIR}/logs/competitive4_parallel_${TS}.log}"
PID_PATH="${PAPER_DIR}/logs/competitive4_parallel_latest.pid"
META_PATH="${PAPER_DIR}/logs/competitive4_parallel_latest.meta"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A competitive4 parallel experiment is already running with PID $old_pid" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
export NUM_TRIALS BUDGET TEMPERATURE MAX_TOKENS WORKERS CHECKPOINT_EVERY OUTPUT LOG_PATH RUN_SCRIPT PYTHONUNBUFFERED=1

pid="$(
python3 - <<'PY'
import os
import subprocess

root = os.getcwd()
cmd = [
    "python3",
    os.environ["RUN_SCRIPT"],
    "--benchmark",
    "paper_competitive4",
    "--num-trials",
    os.environ["NUM_TRIALS"],
    "--budget",
    os.environ["BUDGET"],
    "--temperature",
    os.environ["TEMPERATURE"],
    "--max-tokens",
    os.environ["MAX_TOKENS"],
    "--workers",
    os.environ["WORKERS"],
    "--checkpoint-every",
    os.environ["CHECKPOINT_EVERY"],
    "--resume",
    "--output",
    os.environ["OUTPUT"],
]
with open(os.environ["LOG_PATH"], "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        cmd,
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
output=$OUTPUT
num_trials=$NUM_TRIALS
budget=$BUDGET
temperature=$TEMPERATURE
max_tokens=$MAX_TOKENS
workers=$WORKERS
checkpoint_every=$CHECKPOINT_EVERY
EOF

echo "Started competitive4 parallel background run"
echo "pid: $pid"
echo "log: $LOG_PATH"
echo "output: $OUTPUT"
