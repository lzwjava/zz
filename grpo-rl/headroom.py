#!/usr/bin/env python3
"""
Where does GRPO actually have room to learn?

Measures base-model (and optionally adapter) accuracy per problem category, so a
training tier can be chosen that the base model does *not* already solve. Run 1 used
the 'standard' synthetic mix, where the untouched base model was already ~83-92%
correct -- so the reward saturated (82% zero-variance groups), gradients vanished, and
almost all of the measured "gain" turned out to be the model learning to stop emitting
a stray </think>, not to do arithmetic.

GRPO needs prompt groups with *mixed* outcomes: if all G generations of a prompt get
the same reward, the advantage is zero and that prompt contributes no gradient. Aim for
categories where the base model sits roughly in the 20-70% band.

Usage
-----
    # base model only, 10 problems per category
    .venv/bin/python headroom.py

    # compare the trained adapter category-by-category
    .venv/bin/python headroom.py --adapter ./grpo_qwen15b_math/final

    # include 24 GSM8K test problems and use a different base checkpoint
    .venv/bin/python headroom.py --model Qwen/Qwen2.5-3B --gsm8k 24
"""

from __future__ import annotations

import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_grpo import _extract_answer, make_gsm8k

# Same prompt as training, so accuracy here is directly comparable to the reward there.
SYSTEM = (
    "Solve the math problem. Reason step by step inside <think> ... </think> tags, "
    "then write the final numeric answer right after the closing </think> tag."
)


def build_prompt(problem: str) -> str:
    return f"{SYSTEM}\n\nProblem: {problem}\nSolution:\n"


def build_categories(n: int, seed: int = 5) -> dict[str, list[tuple[str, int]]]:
    """One list of (problem, gold) per arithmetic category, easiest first."""
    kinds = ["add2", "add3", "sub3", "mul1", "mul2", "twoterm", "threeterm"]
    cats: dict[str, list[tuple[str, int]]] = {}

    # Seed per category by index, not by hash(): str hashing is randomised per process,
    # which would make two runs of this script disagree.
    for idx, problem_kind in enumerate(kinds):
        r = random.Random(seed * 1000 + idx)
        rows: list[tuple[str, int]] = []
        for _ in range(n):
            if problem_kind == "add2":
                a, b = r.randint(1, 20), r.randint(1, 20)
                rows.append((f"{a} + {b} = ?", a + b))
            elif problem_kind == "add3":
                a, b = r.randint(100, 999), r.randint(100, 999)
                rows.append((f"{a} + {b} = ?", a + b))
            elif problem_kind == "sub3":
                a, b = r.randint(300, 999), r.randint(100, 299)
                rows.append((f"{a} - {b} = ?", a - b))
            elif problem_kind == "mul1":
                a, b = r.randint(2, 9), r.randint(2, 9)
                rows.append((f"{a} * {b} = ?", a * b))
            elif problem_kind == "mul2":
                a, b = r.randint(12, 99), r.randint(12, 99)
                rows.append((f"{a} * {b} = ?", a * b))
            elif problem_kind == "twoterm":
                a, b, c, d = (r.randint(2, 12) for _ in range(4))
                rows.append((f"{a} * {b} + {c} * {d} = ?", a * b + c * d))
            else:  # threeterm
                a, b, c, d = r.randint(20, 99), r.randint(2, 12), r.randint(2, 12), r.randint(2, 40)
                rows.append((f"{a} + {b} * {c} - {d} = ?", a + b * c - d))
        cats[problem_kind] = rows
    return cats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter to score as well")
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--gsm8k", type=int, default=0, help="number of GSM8K test problems (0 = skip)")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def accuracy(model, tokenizer, pairs, max_new_tokens: int) -> tuple[int, int, float]:
    """Greedy accuracy, plus how often a well-formed <think> block was produced."""
    import re

    ok = tags = 0
    for problem, gold in pairs:
        ids = tokenizer(build_prompt(problem), return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        ok += _extract_answer(text) == str(gold)
        tags += bool(re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL))
    n = len(pairs)
    return ok, n, tags / n if n else 0.0


def verdict(acc: float) -> str:
    if acc >= 0.9:
        return "saturated - no gradient"
    if acc >= 0.75:
        return "little headroom"
    if acc >= 0.25:
        return "GOOD - mixed outcomes"
    return "too hard (all-fail groups)"


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto", "dtype": torch.bfloat16}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"scoring adapter: {args.adapter}")

    cats = build_categories(args.per_category, args.seed)
    if args.gsm8k:
        cats[f"gsm8k ({args.gsm8k})"] = make_gsm8k("test", args.gsm8k)

    print(f"\n=== {args.model} - greedy, {args.per_category} problems/category ===")
    print(f"{'category':<14} {'acc':>7}  {'think':>6}  verdict")
    weakest: list[str] = []
    for name, pairs in cats.items():
        ok, n, tags = accuracy(model, tokenizer, pairs, args.max_new_tokens)
        a = ok / n
        if 0.25 <= a < 0.75:
            weakest.append(name)
        print(f"{name:<14} {ok:>3}/{n:<3} {a:>6.0%}  {tags:>5.0%}  {verdict(a)}")

    print("\nBest training tiers (mixed reward within a group of G):",
          ", ".join(weakest) if weakest else "NONE - everything is saturated or too hard")
    if not weakest:
        print("  -> don't train on this distribution; make the problems harder or use GSM8K")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
