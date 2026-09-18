#!/usr/bin/env python3
"""
Evaluate a GRPO LoRA adapter against its untouched base model.

Reports paired accuracy on the same prompts (the only comparison that actually
answers "did RL help?"), plus how often the model emits well-formed
<think>...</think> blocks -- the formatting objective that the first GRPO run
silently failed to learn.

Modes
-----
  indist : fresh problems from the same generator/distribution as training.
           This is where a working adapter should show a clear gain.
  ood    : deliberately harder problems (3-digit addition, 2-digit
           multiplication, larger parenthesised expressions) drawn from a
           distribution the policy never saw. Measures generalisation, or
           overfitting to the narrow training distribution.

Examples
--------
    # the shipped run, greedy, 64 in-distribution prompts
    .venv/bin/python eval_grpo_adapter.py --mode indist --n 64 --greedy

    # held-out harder problems, sampled (matches GRPO's own rollout temp)
    .venv/bin/python eval_grpo_adapter.py --mode ood --n 32 --temperature 0.7

    # compare two adapters, and dump machine-readable results
    .venv/bin/python eval_grpo_adapter.py --mode indist --save-json results.json

Note
----
The base model is loaded once and evaluated *before* the adapter is attached,
so the baseline is genuinely the untouched policy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
DEFAULT_ADAPTER = HERE / "grpo-rl" / "grpo_qwen15b_math" / "final"

# Must stay byte-identical to train_grpo.py's SYSTEM prompt, or the adapter is
# evaluated off-distribution.
SYSTEM = (
    "Solve the math problem. Reason step by step inside <think> ... </think> tags, "
    "then write the final numeric answer right after the closing </think> tag."
)


def build_prompt(problem: str) -> str:
    return f"{SYSTEM}\n\nProblem: {problem}\nSolution:\n"


# --------------------------------------------------------------------------------------
# Scoring - mirrors train_grpo.py's reward extraction so numbers are comparable
# --------------------------------------------------------------------------------------


def extract_answer(text: str, legacy: bool = False) -> str | None:
    """Number after </think>, falling back to the last number in the text.

    legacy=True reproduces run 1's buggy rule, which took *only* the text after the last
    </think>. A base model often emits a stray trailing </think> with nothing after it, so
    that rule discarded a correct answer and charged the -0.7 'no answer' penalty. On 64
    greedy in-distribution prompts it scored the untouched base model at 82.8% vs 100%
    with the fallback -- i.e. all 11 'failures' were the artefact, and run 1's apparent
    82.8% -> 100% improvement was entirely the policy learning to stop emitting a stray
    tag, not learning any arithmetic. Kept only for reproducing those old numbers.
    """
    after = text.split("</think>")[-1] if "</think>" in text else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", after)
    if not legacy and not nums and "</think>" in text:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)  # stray/unclosed tag: rescue the answer
    if not nums:
        return None
    try:
        return str(int(float(nums[-1])))
    except (ValueError, OverflowError):
        return None


def has_think_block(text: str) -> bool:
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return bool(blocks and len(blocks[0].strip()) >= 20)


# --------------------------------------------------------------------------------------
# Problem sets
# --------------------------------------------------------------------------------------


def make_indist(n: int, seed: int = 1234) -> list[tuple[str, str]]:
    """Same three difficulty tiers as train_grpo.make_synthetic, fresh seed."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for _ in range(n):
        level = rng.choice(["easy", "medium", "hard"])
        if level == "easy":
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            out.append((f"{a} + {b} = ?", str(a + b)))
        elif level == "medium":
            a, b, c = rng.randint(1, 15), rng.randint(1, 10), rng.randint(1, 5)
            out.append((f"{a} * {b} + {c} = ?", str(a * b + c)))
        else:
            a, b, c = rng.randint(2, 12), rng.randint(2, 8), rng.randint(1, 10)
            out.append((f"({a} + {b}) * {c} = ?", str((a + b) * c)))
    return out


def make_ood(n: int, seed: int = 7) -> list[tuple[str, str]]:
    """Harder than anything in training: bigger operands, unseen problem forms."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    for _ in range(n1):
        a, b, c = rng.randint(20, 99), rng.randint(20, 99), rng.randint(3, 12)
        out.append((f"({a} + {b}) * {c} = ?", str((a + b) * c)))
    for _ in range(n2):
        a, b = rng.randint(120, 900), rng.randint(120, 900)
        out.append((f"{a} + {b} = ?", str(a + b)))
    for _ in range(n3):
        a, b = rng.randint(12, 40), rng.randint(12, 40)
        out.append((f"{a} * {b} = ?", str(a * b)))
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="base checkpoint the adapter was trained on")
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="path to the LoRA adapter directory")
    p.add_argument("--mode", default="indist", choices=["indist", "ood"])
    p.add_argument("--n", type=int, default=64, help="number of problems")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--greedy", action="store_true", help="deterministic decoding (do_sample=False)")
    p.add_argument("--temperature", type=float, default=0.7, help="used only when not --greedy")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--legacy-extraction", action="store_true",
                   help="reproduce run 1's buggy '</think>' rule (scores the base model at "
                        "82.8%% instead of 100%% on the training distribution)")
    p.add_argument("--show-samples", type=int, default=3, help="how many graded samples to print")
    p.add_argument("--save-json", default=None, help="optional path to dump full per-problem results")
    return p.parse_args()


def generate(model, tokenizer, problem: str, args) -> str:
    ids = tokenizer(build_prompt(problem), return_tensors="pt").to(model.device)
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    with torch.no_grad():
        out = model.generate(**ids, **kwargs)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(model, tokenizer, problems, args) -> dict:
    model.eval()
    results = []
    for problem, gold in problems:
        text = generate(model, tokenizer, problem, args)
        pred = extract_answer(text, legacy=args.legacy_extraction)
        results.append(
            {
                "problem": problem,
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "think_tags": has_think_block(text),
                "words": len(text.split()),
                "completion": text.strip(),
            }
        )
    n = len(results)
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "think_tag_rate": sum(r["think_tags"] for r in results) / n,
        "mean_words": sum(r["words"] for r in results) / n,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise SystemExit(f"adapter not found: {adapter_path}")

    problems = make_indist(args.n, args.seed) if args.mode == "indist" else make_ood(args.n, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    decoding = "greedy" if args.greedy else f"sampled (T={args.temperature}, top_p={args.top_p})"
    print(f"mode={args.mode}  n={len(problems)}  decoding={decoding}  seed={args.seed}")

    # Baseline first: the adapter mutates the model in place.
    before = evaluate(base, tokenizer, problems, args)
    tuned_model = PeftModel.from_pretrained(base, str(adapter_path))
    after = evaluate(tuned_model, tokenizer, problems, args)

    print(f"\n{'':6} {'accuracy':>12} {'think-tags':>12} {'mean words':>12}")
    for tag, m in (("BASE", before), ("TUNED", after)):
        print(f"{tag:6} {m['accuracy']:>11.1%} {m['think_tag_rate']:>11.1%} {m['mean_words']:>12.0f}")

    fixed = sum(1 for b, a in zip(before["results"], after["results"]) if a["correct"] and not b["correct"])
    broke = sum(1 for b, a in zip(before["results"], after["results"]) if b["correct"] and not a["correct"])
    print(f"\npaired: fixed={fixed}  broken={broke}  net={fixed - broke:+d}")
    print(f"delta accuracy: {after['accuracy'] - before['accuracy']:+.1%}")

    if args.show_samples:
        print("\n--- samples that changed ---")
        shown = 0
        for b, a in zip(before["results"], after["results"]):
            if b["correct"] != a["correct"] and shown < args.show_samples:
                shown += 1
                print(f"  {a['problem']} gold={a['gold']}  base={'OK' if b['correct'] else 'X'}"
                      f" -> tuned={'OK' if a['correct'] else 'X'}")
                print(f"    base : {b['completion'][:180]!r}")
                print(f"    tuned: {a['completion'][:180]!r}")

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "base": before,
                    "tuned": after,
                    "paired": {"fixed": fixed, "broken": broke, "net": fixed - broke},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.save_json}")

    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
