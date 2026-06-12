#!/usr/bin/env python3
"""
Export merged model to GGUF for llama.cpp / ollama.

Usage:
    python export_gguf.py                              # default: q4_k_m
    python export_gguf.py --quant q8_0                 # higher quality
    python export_gguf.py --model ./lzw-notes-merged   # custom path
"""

import argparse
from pathlib import Path

DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DIR / "lzw-notes-merged"))
    p.add_argument("--output", default=str(DIR / "lzw-notes-gguf"))
    p.add_argument("--quant", default="q4_k_m", help="Quantization: q4_k_m, q8_0, f16, etc.")
    return p.parse_args()


def main():
    args = parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

    print(f"Loading merged model from {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        load_in_4bit=False,
    )

    print(f"Exporting to GGUF ({args.quant}) → {args.output}/")
    model.save_pretrained_gguf(args.output, tokenizer, quantization_method=args.quant)

    print(f"\nDone! GGUF files in {args.output}/")
    print(f"  To use with ollama:")
    print(f"    1. Create Modelfile:")
    print(f"       FROM {args.output}/unsloth.Q4_K_M.gguf")
    print(f"       PARAMETER temperature 0.7")
    print(f"       PARAMETER num_ctx 4096")
    print(f"    2. Build: ollama create lzw-notes -f Modelfile")
    print(f"    3. Run:   ollama run lzw-notes")


if __name__ == "__main__":
    main()
