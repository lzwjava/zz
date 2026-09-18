#!/usr/bin/env python3
"""
GRPO RL experiment: teach chain-of-thought arithmetic to a *base* (non-instruct) LLM.

Recipe: base model -> RL directly (no SFT warmup), following DeepSeek-R1 / Qwen thinking-mode.
Designed to fit a single 12 GB RTX 4070 via 4-bit QLoRA + LoRA + gradient checkpointing.

Examples
--------
Smoke test (2 optimizer steps, verifies the whole pipeline):
    .venv/bin/python train_grpo.py --smoke

Real run on the recommended first model:
    .venv/bin/python train_grpo.py --model Qwen/Qwen2.5-1.5B --num-epochs 2

Scale up later (needs gradient checkpointing + 4-bit, fits ~12 GB):
    .venv/bin/python train_grpo.py --model Qwen/Qwen2.5-3B --grad-accum 8

Notes
-----
* Base (non-instruct) checkpoints are the right choice for RL: there is no RLHF prior
  fighting the reward signal, so you can observe the policy actually learning.
* The reward is intentionally split into two logged signals (format + correctness) so you
  can see *which* part of the behaviour is improving.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# --------------------------------------------------------------------------------------
# Prompt format
# --------------------------------------------------------------------------------------

SYSTEM = (
    "Solve the math problem. Reason step by step inside <think> ... </think> tags, "
    "then write the final numeric answer right after the closing </think> tag."
)


def build_prompt(problem: str) -> str:
    """Plain-text prompt (no chat template) - base models have no chat template."""
    return f"{SYSTEM}\n\nProblem: {problem}\nSolution:\n"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def make_synthetic(n: int = 800, seed: int = 42) -> list[dict[str, str]]:
    """Arithmetic with three difficulty levels. Answers are computed without eval()."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for _ in range(n):
        level = rng.choice(["easy", "medium", "hard"])
        if level == "easy":
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            problem, answer = f"{a} + {b} = ?", a + b
        elif level == "medium":
            a, b, c = rng.randint(1, 15), rng.randint(1, 10), rng.randint(1, 5)
            problem, answer = f"{a} * {b} + {c} = ?", a * b + c
        else:
            a, b, c = rng.randint(2, 12), rng.randint(2, 8), rng.randint(1, 10)
            problem, answer = f"({a} + {b}) * {c} = ?", (a + b) * c
        rows.append({"prompt": build_prompt(problem), "answer": str(answer)})
    return rows


def make_gsm8k(split: str, n: int | None) -> list[dict[str, str]]:
    """Optional: real grade-school math word problems (already in the local HF cache)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for ex in ds:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"prompt": build_prompt(ex["question"].strip()), "answer": gold})
    return rows


def build_dataset(name: str, eval_size: int, smoke: bool) -> DatasetDict:
    if name == "synthetic":
        rows = make_synthetic(n=64 if smoke else 800)
        split = Dataset.from_list(rows).train_test_split(
            test_size=max(0.05, eval_size / len(rows)), seed=42
        )
        return split
    if name == "gsm8k":
        train_rows = make_gsm8k("train", n=40 if smoke else 1500)
        test_rows = make_gsm8k("test", n=min(eval_size, 32))
        return DatasetDict(
            {"train": Dataset.from_list(train_rows), "test": Dataset.from_list(test_rows)}
        )
    raise ValueError(f"unknown dataset: {name}")


# --------------------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------------------


def _to_text(completion: Any) -> str:
    """TRL passes plain strings here, but handle conversational (list-of-dicts) too."""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return completion or ""


def _extract_answer(text: str) -> str | None:
    """Prefer the number after </think>; fall back to the last number in the text."""
    after = text.split("</think>")[-1] if "</think>" in text else text
    nums = re.findall(r"-?\d+(?:\.\d+)?", after)
    if not nums:
        return None
    try:
        return str(int(float(nums[-1])))
    except (ValueError, OverflowError):
        return None


def reward_format(completions, **kwargs) -> list[float]:
    """Shaping reward: emit a well-formed <think>...</think> block. Max +0.3."""
    out = []
    for completion in completions:
        text = _to_text(completion)
        r = 0.0
        blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if "<think>" in text and "</think>" in text:
            r += 0.2
            if blocks and len(blocks[0].strip()) >= 20:
                r += 0.1  # non-trivial reasoning, not an empty tag pair
        out.append(r)
    return out


def reward_correctness(completions, answer, **kwargs) -> list[float]:
    """Task reward. +1.0 correct / -0.5 wrong / -0.7 no answer at all."""
    out = []
    for completion, gold in zip(completions, answer):
        pred = _extract_answer(_to_text(completion))
        if pred is None:
            out.append(-0.7)  # emitting think-tags with no number cannot pay off
        elif pred == str(gold).strip():
            out.append(1.0)
        else:
            out.append(-0.5)
    return out


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto", "dtype": torch.bfloat16}

    if args.attn != "sdpa":
        model_kwargs["attn_implementation"] = args.attn

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # ~0.4 GB extra saving
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception as exc:  # e.g. flash-attn not built
        if "attn_implementation" not in model_kwargs:
            raise
        print(f"[warn] attn_implementation={args.attn} failed ({exc}); falling back to sdpa")
        model_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # model / data
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                   help="BASE checkpoint (no -Instruct suffix!), e.g. Qwen/Qwen2.5-3B")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "gsm8k"])
    p.add_argument("--output-dir", default="./grpo_qwen15b_math")
    p.add_argument("--eval-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # memory / precision
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    # GRPO
    p.add_argument("--num-generations", type=int, default=4, help="group size G (must divide effective batch)")
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.02, help="KL penalty to the reference policy")
    p.add_argument("--temperature", type=float, default=1.0, help="exploration temperature")
    p.add_argument("--loss-type", default="grpo")
    p.add_argument("--max-completion-length", type=int, default=384)
    p.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint in output-dir")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (needs wandb installed)")
    p.add_argument("--smoke", action="store_true", help="2-step end-to-end pipeline check")

    args = p.parse_args()
    if args.smoke:
        args.num_epochs = 1.0
        args.max_completion_length = 96
        args.save_steps = 10_000
        args.eval_steps = 10_000
        args.eval_size = 8
        args.grad_accum = 2
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - check your torch install.")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    data = build_dataset(args.dataset, args.eval_size, args.smoke)
    print(f"Dataset '{args.dataset}': train={len(data['train'])} eval={len(data['test'])}")
    print(f"Example prompt:\n{data['train'][0]['prompt']!r}\n gold={data['train'][0]['answer']}")

    model, tokenizer = load_model_and_tokenizer(args)

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if args.eval_steps < 10_000 else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="wandb" if args.wandb else "none",
        run_name=os.path.basename(args.output_dir),
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        reward_funcs=[reward_format, reward_correctness],
        processing_class=tokenizer,
    )

    print("=" * 72)
    print(f"model        : {args.model}  (base, 4-bit={args.load_in_4bit}, LoRA r={args.lora_r})")
    print(f"task         : math reasoning via GRPO from {args.dataset}")
    print(f"G / batch    : {args.num_generations} generations x "
          f"{args.per_device_batch_size * args.grad_accum} prompts per step")
    print(f"output       : {args.output_dir}")
    print("=" * 72)

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"Done - LoRA adapter saved to {args.output_dir}/final")

    # ---- quick qualitative check -------------------------------------------------------
    eval_model = trainer.model
    eval_model.eval()
    for problem in ["(7 + 5) * 3 = ?", "1834 + 596 = ?"]:
        prompt = build_prompt(problem)
        inputs = tokenizer(prompt, return_tensors="pt").to(eval_model.device)
        with torch.no_grad():
            out = eval_model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- {problem}\n{completion.strip()}")


if __name__ == "__main__":
    main()
