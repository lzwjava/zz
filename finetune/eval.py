#!/usr/bin/env python3
"""
Quick eval: compare fine-tuned model vs base on held-out titles.

Usage:
    python eval.py                                # default merged model
    python eval.py --model ./lzw-notes-merged
    python eval.py --model unsloth/Qwen3-8B       # base model (for comparison)
"""

import argparse, json
from pathlib import Path

DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DIR / "lzw-notes-merged"))
    p.add_argument("--eval-data", default=str(DIR / "notes_sft_eval.jsonl"))
    p.add_argument("--n", type=int, default=10, help="Number of examples to eval")
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    # Load eval data
    examples = []
    with open(args.eval_data) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)

    # Pick a diverse sample
    import random
    random.seed(42)
    samples = random.sample(examples, min(args.n, len(examples)))

    print(f"Model: {args.model}")
    print(f"Eval samples: {len(samples)}")
    print("=" * 80)

    # Load with vLLM for fast inference
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85)
        sampling = SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens)

        for i, sample in enumerate(samples):
            title = sample["conversations"][0]["content"]
            ref_answer = sample["conversations"][1]["content"][:300]

            messages = [{"role": "user", "content": title}]
            output = llm.chat(messages, sampling)
            generated = output[0].outputs[0].text

            print(f"\n--- Example {i+1} ---")
            print(f"Q: {title}")
            print(f"\nReference (first 300 chars):\n{ref_answer}...")
            print(f"\nModel output:\n{generated}")
            print()

    except ImportError:
        # Fallback: use transformers directly (slower)
        print("vLLM not installed, using transformers (slower)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )

        for i, sample in enumerate(samples):
            title = sample["conversations"][0]["content"]
            ref_answer = sample["conversations"][1]["content"][:300]

            messages = [{"role": "user", "content": title}]
            input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, do_sample=True)
            generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            print(f"\n--- Example {i+1} ---")
            print(f"Q: {title}")
            print(f"\nReference (first 300 chars):\n{ref_answer}...")
            print(f"\nModel output:\n{generated}")
            print()


if __name__ == "__main__":
    main()
