#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/run/vllm.pid}"
UNIT_FILE="${UNIT_FILE:-$ROOT_DIR/run/vllm.unit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [[ -f "$UNIT_FILE" && -x "$(command -v systemctl)" ]]; then
  UNIT_NAME="$(cat "$UNIT_FILE")"
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    echo "Process: running (unit $UNIT_NAME)"
  else
    echo "Process: not running (unit $UNIT_NAME)"
    exit 1
  fi
elif [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Process: running (PID $PID)"
  else
    echo "Process: stale PID file ($PID)"
    rm -f "$PID_FILE"
    exit 1
  fi
else
  echo "Process: not running (no PID file)"
  exit 1
fi

if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "HTTP: ready"
  exit 0
fi

echo "HTTP: not ready yet"
exit 1
