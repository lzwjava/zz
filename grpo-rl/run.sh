#!/usr/bin/env bash
# Full training run on the 12 GB RTX 4070.
#   ./run.sh smoke   -> 2-step pipeline check
#   ./run.sh         -> real 1.5B base-model run
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "smoke" ]]; then
  exec .venv/bin/python train_grpo.py --smoke
fi

# Notes:
#  * per_device_batch_size * grad_accum (2*4=8) must be divisible by --num-generations (4).
#  * Watch VRAM with: watch -n1 nvidia-smi
#  * If you OOM: drop --max-completion-length to 256 or --num-generations to 2.
#  * If reward plateaus: lower --beta (less KL leash) or raise --temperature.
exec .venv/bin/python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --dataset synthetic \
  --output-dir ./grpo_qwen15b_math \
  --num-epochs 2 \
  --per-device-batch-size 2 \
  --grad-accum 4 \
  --num-generations 4 \
  --max-completion-length 384 \
  --lr 5e-6 \
  --beta 0.02 \
  --temperature 1.0 \
  "$@"
