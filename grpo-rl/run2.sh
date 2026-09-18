#!/usr/bin/env bash
# Second GRPO run -- fixes the two things that went wrong in run 1.
#
# Diagnosed with eval_grpo_adapter.py against grpo_qwen15b_math/final:
#   1. The synthetic mix is saturated and run 1's measured gain was fake. With the
#      </think>-fallback bug fixed, the untouched base model scores 64/64 = 100% greedy on
#      the training distribution (vs 82.8% under run 1's buggy rule), and ~100% on every
#      arithmetic category headroom.py can build, including 2-digit multiplication.
#      So the whole 82.8% -> 100% 'improvement' was the policy learning to stop emitting
#      a stray </think> -- a token habit, not arithmetic. GRPO needs groups with MIXED
#      rewards, so the lever is harder problems, not the same ones for longer.
#      => --difficulty harder (3-digit add/sub, 2-digit products, two-term sums)
#      => 1 epoch, and the best checkpoint is now kept automatically
#      NOTE: check headroom.py FIRST. Greedy accuracy being high does not by itself mean
#      there is no gradient -- what matters is sampled variance at T=1.0 within a group.
#   2. Chain-of-thought never appeared: rewards/reward_format/mean was 0.0 for all 760
#      steps and 0/64 final generations had a well-formed <think> block. A base model
#      cannot discover an exact tag pair from a +/-1.0 correctness reward.
#      => --format-warmup 512 supervised steps first (verified: format reward
#         0.00 -> ~0.30 before RL starts)
#
#   ./run2.sh              -> full run  (~1h; harder data means longer generations)
#   ./run2.sh --num-epochs 2
set -euo pipefail
cd "$(dirname "$0")"

exec .venv/bin/python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B \
  --dataset synthetic \
  --difficulty harder \
  --format-warmup 512 \
  --output-dir ./grpo_qwen15b_math_v2 \
  --num-epochs 1 \
  --per-device-batch-size 2 \
  --grad-accum 4 \
  --num-generations 4 \
  --max-completion-length 384 \
  --eval-size 32 \
  --lr 5e-6 \
  --beta 0.02 \
  --temperature 1.0 \
  "$@"
