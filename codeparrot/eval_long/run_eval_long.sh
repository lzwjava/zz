#!/bin/bash
# Long-prompt evaluation for codeparrot-d12 base model
# Tests the model's ability to handle longer context and generate coherent code
# Output is Markdown (.md) with python syntax-highlighted code blocks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

NANOCHAT_DIR="/mnt/data/nanochat"
cd "$NANOCHAT_DIR"
source .venv/bin/activate

echo "=== Codeparrot-d12: Long-Prompt Evaluation ==="
echo ""

python3 "$SCRIPT_DIR/eval_long.py" "$RESULTS_DIR"

echo ""
echo "=== Done ==="
echo "Results: $RESULTS_DIR/long_prompt_results.md"