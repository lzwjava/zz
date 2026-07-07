#!/usr/bin/env bash
# eval_sft.sh — Run inference on the d12 SFT checkpoint with multiple prompts
set -euo pipefail

NANOCHAT_DIR="/mnt/data/nanochat"
cd "$NANOCHAT_DIR"
source .venv/bin/activate

PROMPTS=(
  "What is gradient descent? Explain it simply."
  "Write a Python function to check if a string is a palindrome."
  "If a train travels at 60 mph for 2 hours and then at 80 mph for 1 hour, what is its average speed?"
  "Write a short poem about machine learning."
  "What are the three main types of machine learning?"
)

echo "==================================="
echo "  d12 SFT Inference Evaluation"
echo "  Model: model_008985 (step 8985)"
echo "  Temp: 0.3, Top-k: 50"
echo "==================================="
echo ""

for prompt in "${PROMPTS[@]}"; do
  echo "-----------------------------------"
  echo "PROMPT: $prompt"
  echo "-----------------------------------"
  python -m scripts.chat_cli \
    --source=sft \
    --model-tag=d12 \
    --step=8985 \
    --temperature=0.3 \
    --top-k=50 \
    --prompt="$prompt" 2>&1 | grep -v "INFO\|WARNING"
  echo ""
  echo ""
done