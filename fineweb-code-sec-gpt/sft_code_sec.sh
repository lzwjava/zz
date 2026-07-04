#!/usr/bin/env bash
# === sft_code_sec.sh — SFT fine-tune the code-sec-fineweb model ===
#
# Takes our pretrained d12 (step 50k) and fine-tunes it on chat conversations:
#   - SmolTalk (460K general chat)
#   - Custom code/SEC tasks (1000 curated examples)
#   - MMLU + GSM8K + SpellingBee for reasoning
#
# Usage:
#   bash sft_code_sec.sh                       # full SFT (~4h)
#   bash sft_code_sec.sh --smoke               # 50-step test
# ============================================================
set -euo pipefail

SMOKE="${1:-}"
NANOCHAT_DIR="/mnt/data/nanochat"
ZZ_DIR="/mnt/data/zz"
SFT_DATA="$ZZ_DIR/datasets/nanochat-mixed/sft_code_sec.jsonl"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Free GPU
for proc in llama-server vllm ollama; do
  pgrep -x "$proc" >/dev/null 2>&1 && pkill -x "$proc" 2>/dev/null && echo "Killed $proc" || true
done
sleep 1

# Step 1: Prepare custom SFT data
if [ ! -f "$SFT_DATA" ]; then
  echo "Preparing custom code+SEC SFT data..."
  python "$ZZ_DIR/fineweb-code-sec-gpt/prepare_sft_data.py" --num 1000 --out "$SFT_DATA"
fi

# Step 2: Ensure identity_conversations.jsonl
IDENTITY="$HOME/.cache/nanochat/identity_conversations.jsonl"
if [ ! -f "$IDENTITY" ]; then
  curl -L -o "$IDENTITY" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# Step 3: Run SFT on the d12 pretrained model
BASE_ARGS=(
  --run=code-sec-sft
  --model-tag=d12
  --model-step=50000
  --tracker=none
  --max-seq-len=2048
  --device-batch-size=4
  --total-batch-size=65536
  --eval-every=500
  --chatcore-every=-1
  --init-lr-frac=0.8
  --warmup-ratio=0.0
  --warmdown-ratio=0.5
  --final-lr-frac=0.0
)

if [ "$SMOKE" = "--smoke" ]; then
  echo "=== SMOKE SFT: 50 steps ==="
  python -m scripts.chat_sft "${BASE_ARGS[@]}" --num-iterations=50 --eval-every=10
else
  echo "=== FULL SFT on d12 (286M) ==="
  echo "  Base: step 50000 (code+SEC+fineweb pretrained)"
  echo "  Data: SmolTalk + CustomJSON ($(wc -l < "$SFT_DATA") code/SEC) + MMLU + GSM8K + Spelling"
  echo "  Steps: full epoch (~1M conversations)"
  python -m scripts.chat_sft "${BASE_ARGS[@]}"
fi