#!/usr/bin/env python3
"""Generate samples from the trained checkpoint using saved prompt files.

Usage:
  python run_prompts.py                          # default: last checkpoint, temp 0.8, 150 tokens
  python run_prompts.py --step 50000 --temp 0.6 --max-tokens 100
  python run_prompts.py --prompt-dir /mnt/data/zz/fineweb-code-sec-gpt
"""
import sys
sys.path.insert(0, "/mnt/data/nanochat")

import argparse, json, glob, os, torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=50000)
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--temp", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=150)
parser.add_argument("--prompt-dir", default="/mnt/data/zz/fineweb-code-sec-gpt")
parser.add_argument("--out", default="/mnt/data/zz/fineweb-code-sec-gpt/results.txt")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = get_tokenizer()
n_embd = args.depth * 64
n_head = n_embd // 128
cfg = GPTConfig(
    sequence_len=2048, vocab_size=32768,
    n_layer=args.depth, n_head=n_head, n_kv_head=n_head,
    n_embd=n_embd, window_pattern="L",
)
model = GPT(cfg).to(device)
checkpoint_dir = f"/home/lzw/.cache/nanochat/base_checkpoints/d{args.depth}"
model_data, _, _ = load_checkpoint(checkpoint_dir, args.step, device, load_optimizer=False, rank=0)
model.load_state_dict(model_data, strict=True)
model.eval()
engine = Engine(model, tokenizer)

prompt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
prompt_files = [f for f in prompt_files if "results" not in f and "README" not in f]

print(f"Found {len(prompt_files)} prompt files\n")

all_results = []

for pf in prompt_files:
    with open(pf) as f:
        prompt = f.read().strip()
    basename = os.path.basename(pf).replace(".txt", "")
    print(f"--- {basename} ---")

    # Run at multiple lengths
    for max_tok in [80, 200]:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with torch.no_grad():
            sample, _ = engine.generate_batch(
                tokens, num_samples=1,
                max_tokens=max_tok,
                temperature=args.temp,
                top_k=args.top_k,
            )
        text = tokenizer.decode(sample[0])

        domain = basename.split("_")[0]
        result = {
            "prompt": basename,
            "domain": domain,
            "temperature": args.temp,
            "max_tokens": max_tok,
            "input_len": len(tokens[0]) if hasattr(tokens[0], '__len__') else len(tokens),
            "output_len": len(sample[0]),
            "output": text,
        }
        all_results.append(result)

        # Print
        print(f"  [{max_tok} tok | temp={args.temp}]")
        print(f"  Input:  {prompt[:80]}...")
        print(f"  Output: {text[:200]}...")
        print()

# Save results
with open(args.out, "w") as f:
    f.write(f"=== Model: d{args.depth} step {args.step} temp={args.temp} ===\n\n")
    for r in all_results:
        f.write(f"--- {r['prompt']} ({r['max_tokens']} tokens) ---\n")
        f.write(f"Input tokens: {r['input_len']} | Output tokens: {r['output_len']}\n")
        f.write(r["output"] + "\n\n")

print(f"\nResults saved to {args.out}")
print(f"Total generations: {len(all_results)}")