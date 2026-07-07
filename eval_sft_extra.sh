#!/usr/bin/env bash
# eval_sft_extra.sh — Additional focused inference tests
set -euo pipefail

NANOCHAT_DIR="/mnt/data/nanochat"
cd "$NANOCHAT_DIR"
source .venv/bin/activate

PROMPTS=(
  "What is 2+2?"
  "What comes after 1, 1, 2, 3, 5, 8?"
  "Explain what a neural network is in one sentence."
  "Write a SQL query to find all employees hired in 2023."
  "Capital of France?"
)

for prompt in "${PROMPTS[@]}"; do
  echo "-----------------------------------"
  echo "PROMPT: $prompt"
  echo "-----------------------------------"
  python -m scripts.chat_cli \
    --source=sft \
    --model-tag=d12 \
    --step=8985 \
    --temperature=0.1 \
    --top-k=10 \
    --prompt="$prompt" 2>&1 | grep -v "INFO\|WARNING\|✓\|Autodetected\|NanoChat Interactive\|Type 'quit\|Type 'clear\|-------"
  echo ""
  echo ""
done