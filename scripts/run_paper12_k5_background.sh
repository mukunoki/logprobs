#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_paper12_k5_all.sh"
cd "$REPO_ROOT"

mkdir -p "${PAPER_DIR}/logs" "${RESULTS_DIR}"

NUM_TRIALS="${NUM_TRIALS:-5}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/paper12_k5_${TS}}"
LOG_PATH="${LOG_PATH:-${PAPER_DIR}/logs/paper12_k5_${TS}.log}"
PID_PATH="${PAPER_DIR}/logs/paper12_k5_latest.pid"
META_PATH="${PAPER_DIR}/logs/paper12_k5_latest.meta"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A paper12_k5 experiment is already running with PID $old_pid" >&2
    echo "If you need to resume another run, stop that process first or remove $PID_PATH after confirming it is stale." >&2
    exit 1
  fi
fi

export NUM_TRIALS BUDGET TEMPERATURE CHECKPOINT_EVERY OUT_DIR LOG_PATH RUN_SCRIPT PYTHONUNBUFFERED=1

pid="$(
python3 - <<'PY'
import os
import subprocess

root = os.getcwd()
log_path = os.environ["LOG_PATH"]
run_script = os.environ["RUN_SCRIPT"]
env = os.environ.copy()
with open(log_path, "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["bash", run_script],
        cwd=root,
        env=env,
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
num_trials=$NUM_TRIALS
budget=$BUDGET
temperature=$TEMPERATURE
checkpoint_every=$CHECKPOINT_EVERY
EOF

echo "Started paper12_k5 background run"
echo "pid: $pid"
echo "log: $LOG_PATH"
echo "out_dir: $OUT_DIR"
