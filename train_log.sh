#!/usr/bin/env bash
# ============================================================
# train_log.sh — monitor & extract training metrics
# ============================================================
# Usage:
#   bash train_log.sh                              # latest train.log tail
#   bash train_log.sh --watch                      # live follow
#   bash train_log.sh --summary                    # extract bpb/loss curve
#   bash train_log.sh --last                       # last run's train.log path
# ============================================================
set -euo pipefail

NANOCHAT_DIR="/mnt/data/nanochat"
MODE="${1:-tail}"

LOG_DIR="$HOME/.cache/nanochat/base_checkpoints"

case "$MODE" in
  --watch|-w)
    latest=$(ls -t "$LOG_DIR"/*/train.log 2>/dev/null | head -1)
    if [ -z "$latest" ]; then echo "No train.log found"; exit 1; fi
    echo "Following: $latest"
    tail -f "$latest"
    ;;
  --summary|-s)
    latest=$(ls -t "$LOG_DIR"/*/train.log 2>/dev/null | head -1)
    if [ -z "$latest" ]; then echo "No train.log found"; exit 1; fi
    echo "=== Summary: $latest ==="
    grep -E "Validation bpb|Step |Peak memory|Total training|train/loss" "$latest" | tail -30
    ;;
  --last|-l)
    latest=$(ls -t "$LOG_DIR"/*/train.log 2>/dev/null | head -1)
    if [ -z "$latest" ]; then echo "No train.log found"; exit 1; fi
    echo "$latest"
    ;;
  *)
    latest=$(ls -t "$LOG_DIR"/*/train.log 2>/dev/null | head -1)
    if [ -z "$latest" ]; then echo "No train.log found"; exit 1; fi
    echo "=== Train log: $latest === (use --watch to follow)"
    tail -20 "$latest"
    ;;
esac