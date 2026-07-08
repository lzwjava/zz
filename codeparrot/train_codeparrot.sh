#!/bin/bash
# Train nanochat on codeparrot-clean (24.7B Python tokens, 247 shards)
# RTX 4070 (12 GB VRAM), single GPU
#
# Model: depth=12 (~286M params), batch=65536, seq=2048
# Data: ~24.7B tokens available
#
# Usage:
#   bash train_codeparrot.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DATA_DIR="/mnt/data/zz/datasets/codeparrot-clean-nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

cd /mnt/data/nanochat
source .venv/bin/activate

echo "=== Codeparrot-clean pretraining on RTX 4070 ==="
echo "Model:  286M params (depth=12, dim=768, heads=6)"
echo "Data:   24.7B Python tokens, 54 parquet shards"
echo ""

# Step 1: Train tokenizer on codeparrot data
echo "=== Step 1: Train tokenizer on codeparrot code ==="
python -m scripts.tok_train \
    --max-chars=2000000000 \
    --vocab-size=32768

# Step 2: Pretrain
echo ""
echo "=== Step 2: Pretrain base model ==="
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --total-batch-size=65536 \
    --max-seq-len=2048 \
    --window-pattern L \
    --num-iterations=87000 \
    --eval-every=2000 \
    --eval-tokens=524288 \
    --sample-every=5000 \
    --save-every=10000 \
    --core-metric-every=-1 \
    --run=codeparrot-d12 \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Evaluate: python -m scripts.base_eval --device-batch-size=8"
echo "Chat:     python -m scripts.chat_cli -p 'Write a Python function to reverse a linked list'"
echo "Web UI:   python -m scripts.chat_web"