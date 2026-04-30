#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
RESULTS_DIR="${REPO_ROOT}/results"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_paper12_k5_all.sh"
cd "$REPO_ROOT"

export NUM_TRIALS="${NUM_TRIALS:-5}"
export BUDGET="${BUDGET:-5}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
export PAPER_EASY4_MAX_TOKENS="${PAPER_EASY4_MAX_TOKENS:-4096}"
export PAPER_MEDIUM4_MAX_TOKENS="${PAPER_MEDIUM4_MAX_TOKENS:-4096}"
export PAPER_HARD4_MAX_TOKENS="${PAPER_HARD4_MAX_TOKENS:-4096}"

TS="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/paper12_cbba_${TS}}"
export FIGURE_OUT_DIR="${FIGURE_OUT_DIR:-${PAPER_DIR}}"

exec bash "$RUN_SCRIPT"
