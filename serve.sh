#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
if [[ ! -f "$VENV_DIR/bin/activate" && -f "/home/mukunoki/llminf/.venv/bin/activate" ]]; then
  VENV_DIR="/home/mukunoki/llminf/.venv"
fi
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing venv: $VENV_DIR/bin/activate" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

HF_HOME="${HF_HOME:-$ROOT_DIR/hf-cache}"
if [[ "$HF_HOME" == "$ROOT_DIR/hf-cache" && -d "/home/mukunoki/llminf/hf-cache" ]]; then
  HF_HOME="/home/mukunoki/llminf/hf-cache"
fi
export HF_HOME

MODEL="${MODEL:-Ankushbl6/Qwen3.5-35B-A3B-AWQ-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

mkdir -p "$HF_HOME"

MIN_FREE_GB="${MIN_FREE_GB:-5}"
AVAIL_KB="$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')"
if [[ -n "$AVAIL_KB" ]]; then
  AVAIL_GB="$((AVAIL_KB / 1024 / 1024))"
  if (( AVAIL_GB < MIN_FREE_GB )); then
    echo "Not enough disk space for HF cache: $AVAIL_GB GB available (< ${MIN_FREE_GB} GB)" >&2
    echo "Set HF_HOME to a filesystem with more space and retry." >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  EXTRA_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}"
