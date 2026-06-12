#!/usr/bin/env python3
"""
SFT training — pure transformers + peft (no Unsloth Triton).

Usage:
    python train.py                    # default: Qwen3-4B, 4-bit, 2 epochs
    python train.py --epochs 3         # more epochs
    python train.py --batch-size 4     # larger batch
    python train.py --max-steps 100    # quick test run
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DIR = Path(__file__).parent

DEFAULT_MODEL = "/home/lzw/.cache/huggingface/hub/models--unsloth--Qwen3-4B-unsloth-bnb-4bit/snapshots/main"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    p.add_argument("--data", default=str(DIR / "notes_sft_train.jsonl"))
    p.add_argument("--eval-data", default=str(DIR / "notes_sft_eval.jsonl"))
    p.add_argument("--output", default=str(DIR / "lzw-notes-lora"))
    p.add_argument("--merged-output", default=str(DIR / "lzw-notes-merged"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--packing", action="store_true", default=False)
    p.add_argument("--no-packing", dest="packing", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Model:       {args.model}")
    print(f"LoRA r:      {args.lora_r}")
    print(f"Batch size:  {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print()

    # Load model
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("Loading dataset...", flush=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds_eval = load_dataset("json", data_files=args.eval_data, split="train")

    # Apply chat template
    ds = ds.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False)},
        remove_columns=ds.column_names,
    )
    ds_eval = ds_eval.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["conversations"], tokenize=False)},
        remove_columns=ds_eval.column_names,
    )

    print(f"Train: {len(ds)}, Eval: {len(ds_eval)}", flush=True)

    # Train
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=ds_eval,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            output_dir=args.output,
            packing=args.packing,
            seed=42,
            report_to="none",
        ),
    )

    print("Starting training...", flush=True)
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output}/", flush=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\nDone!", flush=True)
    print(f"  LoRA adapter:  {args.output}/")
    print(f"  To merge:      python merge.py")
    print(f"  To GGUF:       python export_gguf.py")
    print(f"  To serve:      vllm serve {args.output} --max-model-len 4096")


if __name__ == "__main__":
    main()
