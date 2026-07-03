#!/usr/bin/env bash
set -euo pipefail
# === fineweb-code-sec-gpt  —  mixed-data nanochat pretraining ===
# Trains a d12 (286M) model on github-code + sec-edgar + fineweb-edu.
#
# Usage:
#   bash /mnt/data/zz/fineweb-code-sec-gpt.sh          # full 50k-step run
#   bash /mnt/data/zz/fineweb-code-sec-gpt.sh --smoke   # 200-step smoke test

SMOKE="${1:-}"
MODE="${SMOKE:---full}"

NANOCHAT_DIR="/mnt/data/nanochat"
export NANOCHAT_DATA_DIR="/mnt/data/zz/datasets/nanochat-mixed"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

BASE_ARGS=(
  --depth=12
  --device-batch-size=8
  --total-batch-size=65536
  --max-seq-len=2048
  --window-pattern=L
  --eval-every=2000
  --sample-every=5000
  --save-every=5000
  --tracker=none
  --run=code-sec-fineweb-d12
)

if [ "$MODE" = "--smoke" ]; then
  echo "=== SMOKE TEST: 200 steps ==="
  python -m scripts.base_train \
    "${BASE_ARGS[@]}" \
    --num-iterations=200 \
    --eval-every=100 \
    --sample-every=-1 \
    --save-every=-1
else
  echo "=== FULL TRAIN: 50,000 steps ==="
  echo "  Data: $NANOCHAT_DATA_DIR (64 parquets: code + sec-edgar + fineweb-edu)"
  echo "  Tokens: 3.28B  |  Tokens:param ratio: 29.8"
  echo "  Est time: ~16.5 hours"
  python -m scripts.base_train "${BASE_ARGS[@]}" --num-iterations=50000
fi