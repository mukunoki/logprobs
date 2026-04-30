#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/run/vllm.pid}"
UNIT_FILE="${UNIT_FILE:-$ROOT_DIR/run/vllm.unit}"

if [[ -f "$UNIT_FILE" && -x "$(command -v systemctl)" ]]; then
  UNIT_NAME="$(cat "$UNIT_FILE")"
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    systemctl --user stop "$UNIT_NAME"
    echo "Stopped (unit $UNIT_NAME)"
  else
    echo "Unit not running ($UNIT_NAME)"
  fi
  rm -f "$UNIT_FILE"
  exit 0
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process not running (PID $PID)"
  rm -f "$PID_FILE"
  exit 0
fi

kill "$PID"
for _ in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped (PID $PID)"
    exit 0
  fi
  sleep 1
done

echo "Failed to stop PID $PID" >&2
exit 1
