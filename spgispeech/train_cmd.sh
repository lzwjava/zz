#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Train whisper-small on SPGISpeech S — ready-to-run in tmux
#
# Usage:
#   cd /mnt/data/zz/spgispeech && source .venv/bin/activate
#   nohup bash train_cmd.sh > train_small.log 2>&1 &
#
# Or in tmux:
#   tmux new-session -s whisper \; send-keys 'cd /mnt/data/zz/spgispeech && source .venv/bin/activate && bash train_cmd.sh' Enter
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

TRAIN_CMD=(
    python3 train_whisper.py
    --model small
    --batch-size 16
    --grad-accum 2
    --lr 1e-5
    --epochs 3
    --save-steps 500
    --eval-steps 500
    --logging-steps 50
    --eval-samples 500
    --test-samples 2000
    --max-audio-sec 30.0
)

echo "═══ Starting: ${TRAIN_CMD[*]} ═══"
echo "Started at: $(date)"
echo ""

"${TRAIN_CMD[@]}"

echo ""
echo "Finished at: $(date)"
