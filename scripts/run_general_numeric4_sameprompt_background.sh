#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
LOG_DIR="${REPO_ROOT}/paper/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

NUM_TRIALS="${NUM_TRIALS:-50}"
BUDGET="${BUDGET:-5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
VLLM_USE_CHAT="${VLLM_USE_CHAT:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/general_numeric4_sameprompt_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/general_numeric4_sameprompt_${TS}_${NUM_TRIALS}trials_w${WORKERS}_mt${MAX_TOKENS}.log}"
PID_PATH="${PID_PATH:-${LOG_PATH%.log}.pid}"
META_PATH="${META_PATH:-${LOG_PATH%.log}.meta}"
RUN_SCRIPT="${SCRIPT_DIR}/run_general_numeric4_sameprompt_all.sh"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[INFO] job already running pid=$old_pid"
    echo "[INFO] pid: $PID_PATH"
    echo "[INFO] log: $LOG_PATH"
    echo "[INFO] out: $OUT_DIR"
    exit 0
  fi
fi

export NUM_TRIALS BUDGET TEMPERATURE MAX_TOKENS WORKERS CHECKPOINT_EVERY VLLM_USE_CHAT OUT_DIR

python3 - <<PY
from pathlib import Path
import os
import subprocess
import time

repo = Path("${REPO_ROOT}")
run_script = Path("${RUN_SCRIPT}")
log_path = Path("${LOG_PATH}")
pid_path = Path("${PID_PATH}")
meta_path = Path("${META_PATH}")
out_dir = Path("${OUT_DIR}")
env = os.environ.copy()

with log_path.open("ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["bash", str(run_script)],
        cwd=str(repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

pid_path.write_text(f"{proc.pid}\\n")
meta_path.write_text(
    "\\n".join([
        f"pid={proc.pid}",
        f"started_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
        "benchmark=paper_general_numeric4",
        f"num_trials={env.get('NUM_TRIALS')}",
        f"budget={env.get('BUDGET')}",
        f"temperature={env.get('TEMPERATURE')}",
        f"max_tokens={env.get('MAX_TOKENS')}",
        f"workers={env.get('WORKERS')}",
        f"vllm_use_chat={env.get('VLLM_USE_CHAT')}",
        f"log={log_path}",
        f"out_dir={out_dir}",
    ])
    + "\\n"
)
print(proc.pid)
PY

echo "[INFO] started numeric4 same-prompt evaluation"
echo "[INFO] pid: $PID_PATH"
echo "[INFO] log: $LOG_PATH"
echo "[INFO] meta: $META_PATH"
echo "[INFO] out: $OUT_DIR"
