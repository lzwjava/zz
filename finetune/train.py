#!/usr/bin/env python3
"""
SFT training with Unsloth — QLoRA on RTX 4070 (12GB).

Usage:
    python train.py                    # default: unsloth/Qwen3-8B, 4-bit
    python train.py --model unsloth/Qwen3-4B    # smaller model
    python train.py --epochs 3         # more epochs
    python train.py --no-4bit          # 16-bit (needs >24GB VRAM)
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3-8B", help="Model name or path")
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
    p.add_argument("--no-4bit", action="store_true", help="Use 16-bit instead of 4-bit QLoRA")
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no-packing", dest="packing", action="store_false")
    return p.parse_args()


def format_conversations(example, tokenizer):
    """Apply chat template to conversations."""
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    args = parse_args()
    use_4bit = not args.no_4bit

    print(f"Model:       {args.model}")
    print(f"4-bit:       {use_4bit}")
    print(f"LoRA r:      {args.lora_r}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Batch size:  {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Epochs:      {args.epochs}")
    print(f"LR:          {args.lr}")
    print(f"Packing:     {args.packing}")
    print()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=use_4bit,
        dtype=None,  # auto
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,  # alpha = r is a solid default
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC
    )

    # Load dataset
    ds = load_dataset("json", data_files=args.data, split="train")
    ds_eval = load_dataset("json", data_files=args.eval_data, split="train")

    # Apply chat template
    ds = ds.map(lambda x: format_conversations(x, tokenizer), remove_columns=ds.column_names)
    ds_eval = ds_eval.map(lambda x: format_conversations(x, tokenizer), remove_columns=ds_eval.column_names)

    print(f"Train examples: {len(ds)}")
    print(f"Eval examples:  {len(ds_eval)}")
    print(f"Sample:\n{ds[0]['text'][:500]}...\n")

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=ds_eval,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            output_dir=args.output,
            packing=args.packing,
            max_seq_length=args.max_seq_len,
            seed=42,
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output}/")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save merged model (full weights, ready for vLLM/serving)
    print(f"Saving merged model to {args.merged_output}/")
    model.save_pretrained_merged(args.merged_output, tokenizer, save_method="merged_16bit")

    print("\nDone! Next steps:")
    print(f"  1. Serve:  vllm serve {args.merged_output} --max-model-len {args.max_seq_len}")
    print(f"  2. GGUF:   python export_gguf.py")
    print(f"  3. Eval:   python eval.py")


if __name__ == "__main__":
    main()
