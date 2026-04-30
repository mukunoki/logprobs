#!/usr/bin/env bash
set -euo pipefail

# Run all shared-candidate CBBA experiments in a resumable, checkpointed way.
# Intended for nohup/background execution; do not wrap this script with timeout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
RESULT_DIR="${REPO_ROOT}/results"
LOG_DIR="${PAPER_DIR}/logs"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

LOCK_FILE="${LOG_DIR}/shared_candidate_offline_all.lock"
if [[ -e "$LOCK_FILE" ]]; then
  old_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Another shared-candidate experiment is already running with PID $old_pid" >&2
    exit 1
  fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

export PYTHONUNBUFFERED=1

run_one() {
  local benchmark="$1"
  local output="${RESULT_DIR}/shared_candidate_offline_eval_${benchmark}.json"

  echo "================================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting ${benchmark}"
  echo "output: ${output}"
  echo "================================================================"

  python3 "${SCRIPTS_DIR}/shared_candidate_offline_eval.py" \
    --benchmark "$benchmark" \
    --num-trials 5 \
    --k 3 \
    --temperature 0.8 \
    --max-tokens 512 \
    --random-repeats 100 \
    --checkpoint-every 1 \
    --resume \
    --output "$output"

  echo "================================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') finished ${benchmark}"
  echo "================================================================"
}

run_one paper12

python3 - <<'PY'
import json
from pathlib import Path
for path in [
    Path('results/shared_candidate_offline_eval_paper12.json'),
]:
    data = json.loads(path.read_text())
    total = data['num_trials'] * data['num_problems']
    done = len(data.get('candidate_sets', []))
    print(f"{path}: {done}/{total}, is_partial={data.get('is_partial')}")
    for key in ['full_topk_best_time', 'generation_order_early_stop', 'confidence_order_early_stop', 'random_order_early_stop']:
        print(key, data['summary'].get(key))
PY
