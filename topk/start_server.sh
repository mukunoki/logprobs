#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/run/vllm.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/run/vllm.log}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WAIT_SECS="${WAIT_SECS:-240}"
USE_SYSTEMD="${USE_SYSTEMD:-1}"
UNIT_NAME="${UNIT_NAME:-vllm-topk}"
UNIT_FILE="${UNIT_FILE:-$ROOT_DIR/run/vllm.unit}"

mkdir -p "$(dirname "$PID_FILE")"

if [[ "$USE_SYSTEMD" == "1" && -x "$(command -v systemd-run)" ]]; then
  if [[ -f "$UNIT_FILE" ]]; then
    UNIT_NAME="$(cat "$UNIT_FILE")"
  fi
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    echo "Server is already running with unit $UNIT_NAME"
    exit 0
  fi
else
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
      echo "Server is already running with PID $PID"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi
fi

if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "An API server is already responding at http://$HOST:$PORT"
  exit 0
fi

if [[ "$USE_SYSTEMD" == "1" && -x "$(command -v systemd-run)" ]]; then
  systemd-run --user --unit "$UNIT_NAME" \
    --property=WorkingDirectory="$ROOT_DIR" \
    --property="StandardOutput=append:$LOG_FILE" \
    --property="StandardError=append:$LOG_FILE" \
    "$ROOT_DIR/serve.sh" >/dev/null
  echo "$UNIT_NAME" >"$UNIT_FILE"
else
  nohup "$ROOT_DIR/serve.sh" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "$PID" >"$PID_FILE"
fi

for _ in $(seq 1 $((WAIT_SECS / 2))); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    if [[ -f "$UNIT_FILE" ]]; then
      echo "Started vLLM server with unit $(cat "$UNIT_FILE")"
    else
      echo "Started vLLM server with PID $PID"
    fi
    echo "Log: $LOG_FILE"
    exit 0
  fi

  if [[ -f "$UNIT_FILE" ]]; then
    if ! systemctl --user is-active --quiet "$(cat "$UNIT_FILE")"; then
      echo "Failed to start server. Check log: $LOG_FILE" >&2
      if [[ -f "$LOG_FILE" ]]; then
        echo "---- Log tail ----" >&2
        tail -n 60 "$LOG_FILE" >&2
        echo "------------------" >&2
      fi
      exit 1
    fi
  elif ! kill -0 "$PID" 2>/dev/null; then
    echo "Failed to start server. Check log: $LOG_FILE" >&2
    if [[ -f "$LOG_FILE" ]]; then
      echo "---- Log tail ----" >&2
      tail -n 60 "$LOG_FILE" >&2
      echo "------------------" >&2
    fi
    rm -f "$PID_FILE"
    exit 1
  fi

  sleep 2
done

echo "Server process is running but HTTP endpoint is not ready yet"
echo "Log: $LOG_FILE"
