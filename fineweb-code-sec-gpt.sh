#!/usr/bin/env bash
# ============================================================
# fineweb-code-sec-gpt.sh — nanochat pretraining on mixed data
# ============================================================
# Trains a d12 (286M params) transformer on:
#   - github-code (38 shards, ~3.9M code files)
#   - sec-edgar   (17 10-K filings)
#   - fineweb-edu (9 shards, ~4.3B tokens from CommonCrawl)
#
# Data pipeline (run once):
#   bash /mnt/data/zz/prepare_data.sh
#
# Usage:
#   bash /mnt/data/zz/fineweb-code-sec-gpt.sh           # full 50k steps (~16h)
#   bash /mnt/data/zz/fineweb-code-sec-gpt.sh --smoke    # 200-step smoke test
# ============================================================
set -euo pipefail

SMOKE="${1:-}"
MODE="${SMOKE:---full}"

NANOCHAT_DIR="/mnt/data/nanochat"
export NANOCHAT_DATA_DIR="/mnt/data/zz/datasets/nanochat-mixed"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Free GPU memory — kill background LLM servers that consume VRAM
for proc in llama-server vllm ollama; do
  if pgrep -x "$proc" >/dev/null 2>&1; then
    echo "  Killing $proc to free GPU memory..."
    pkill -x "$proc" 2>/dev/null || true
    sleep 2
  fi
done

BASE_ARGS=(
  --depth=12
  --device-batch-size=8
  --total-batch-size=65536
  --max-seq-len=2048
  --window-pattern=L
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
  echo "  Data: $NANOCHAT_DATA_DIR"
  echo "  Model: d12 (286M params), seq=2048"
  echo "  Batch: 65,536 tokens/step, grad_accum=4"
  echo "  Total tokens: 3.28B (ratio 29.8x params)"
  echo "  Est time: ~16.5 hours @ 0.84 steps/sec"
  echo ""
  python -m scripts.base_train \
    "${BASE_ARGS[@]}" \
    --num-iterations=50000 \
    --eval-every=2000 \
    --sample-every=5000 \
    --save-every=5000
fi