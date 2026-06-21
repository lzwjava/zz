# GPT-2 124M config — for single RTX 4070 12GB, 14B tokens (github-code)
# Architecture: n_layer=12, n_head=12, n_embd=768 → ~124M params
# Run: cd /mnt/data/nanoGPT && python3 train.py config/train_github_code_124m.py

out_dir = 'out-github-code-124m'
shard_dir = '/mnt/data/zz/datasets/github-code-tok'
dataset = 'github-code'

eval_interval = 1000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = 'github-code-124m'
wandb_run_name = 'gpt2-124m-14B'

# Batch: 4 × 1024 = 4,096 tokens/step (micro-step)
# grad_accum=8 → effective batch = 32,768 tokens/step
# Total: ~427,000 steps × 32,768 = 14B tokens
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 8

# Model — GPT-2 124M
# ~124M params: 12*12*768² + vocab*768
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Optimizer — GPT-3 style
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
max_iters = 427000
lr_decay_iters = 427000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Speed
compile = True
