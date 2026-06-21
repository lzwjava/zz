#!/usr/bin/env python3
"""
Merge LoRA adapter into base model.

Usage:
    python merge.py
    python merge.py --adapter ./lzw-notes-lora --base unsloth/Qwen3-4B --output ./lzw-notes-merged
"""

import argparse
from pathlib import Path

DIR = Path(__file__).parent

DEFAULT_BASE = "/home/lzw/.cache/huggingface/hub/models--unsloth--Qwen3-4B-unsloth-bnb-4bit/snapshots/main"
# Fallback: try the FP16 base if 4-bit doesn't merge well
DEFAULT_BASE_FP16 = "unsloth/Qwen3-4B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE_FP16, help="Base model name or path")
    p.add_argument("--adapter", default=str(DIR / "lzw-notes-lora"))
    p.add_argument("--output", default=str(DIR / "lzw-notes-merged"))
    return p.parse_args()


def main():
    args = parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading base model: {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    print(f"Loading LoRA adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging LoRA weights into base model...", flush=True)
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}/", flush=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone! Merged model saved to {args.output}/")
    print(f"  Next: python export_gguf.py")


if __name__ == "__main__":
    main()
