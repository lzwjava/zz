#!/bin/bash
# Evaluate the codeparrot-d12 model
# Results go to ./results/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

NANOCHAT_DIR="/mnt/data/nanochat"
cd "$NANOCHAT_DIR"
source .venv/bin/activate

echo "=== Codeparrot-d12 Evaluation ==="
echo "Model: 286M params (depth=12), step 87,000"
echo "Data:  codeparrot-clean (Python)"
echo "Val bpb (from training): 0.372"
echo ""

# --- Step 1: BPB eval (quick, ~2 min) ---
echo "=== BPB Evaluation ==="
python -m scripts.base_eval \
    --model-tag d12 \
    --eval bpb \
    --device-batch-size=8 \
    --split-tokens=1048576 \
    2>&1 | tee "$RESULTS_DIR/bpb.log"

# --- Step 2: Sample eval ---
echo ""
echo "=== Sample Generation (built-in prompts) ==="
python -m scripts.base_eval \
    --model-tag d12 \
    --eval sample \
    --device-batch-size=8 \
    2>&1 | tee "$RESULTS_DIR/samples.log"

# --- Step 3: Python-specific code generation ---
echo ""
echo "=== Python Code Generation ==="
python3 -c "
import sys, os
sys.path.insert(0, '$NANOCHAT_DIR')
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.engine import Engine
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model
model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d12')
print(f'Loaded model step {meta[\"step\"]}, val_bpb: {meta[\"val_bpb\"]:.4f}')
engine = Engine(model, tokenizer)

# Python code prompts
prompts = [
    'def fibonacci(n):',
    'def reverse_linked_list(head):',
    'def sort_dict_by_value(d):',
    'class Stack:',
    'import numpy as np',
    '# A function to read a CSV file and return a list of dicts',
    'def is_palindrome(s):',
    '# Binary search implementation',
]

with torch.no_grad():
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend='<|bos|>')
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=128, temperature=0.3)
        sample_str = tokenizer.decode(sample[0])
        print('=' * 70)
        print(sample_str)
" 2>&1 | tee "$RESULTS_DIR/python_samples.log"

echo ""
echo "=== Done ==="
echo "Results in: $RESULTS_DIR"
echo "  bpb.log          - bits-per-byte on train/val split"
echo "  samples.log      - built-in sample output"
echo "  python_samples.log - Python code generation"
"