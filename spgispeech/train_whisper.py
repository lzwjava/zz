#!/usr/bin/env python3
"""Fine-tune Whisper on SPGISpeech S config — robust pyarrow-backed Dataset.

No HF `datasets` Audio feature (torchcodec dependency hell). Uses raw pyarrow
for column access + soundfile for WAV decoding.

Usage:
    python3 train_whisper.py                           # whisper-small
    python3 train_whisper.py --model tiny --epochs 1   # fast test
    python3 train_whisper.py --model medium            # medium
    python3 train_whisper.py --resume                  # resume from checkpoint
"""

import argparse, os, io, sys, json, time, gc
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from evaluate import load as load_metric

DATA_DIR = "/mnt/data/zz/spgispeech/data/S"
OUTPUT_DIR = "/mnt/data/zz/spgispeech/checkpoints"
LOG_FILE = "/mnt/data/zz/spgispeech/train_log.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Custom Dataset (pyarrow-backed, no HF datasets Audio) ───────────────────

class SPGISpeechDataset(Dataset):
    """Torch Dataset over parquet shards. Each shard has ~13 row groups, each
    row group holds ~1000 samples. We build an index mapping (shard, row_group, offset)
    for O(1) random access without loading full shards into RAM."""

    def __init__(self, shard_paths, processor, max_seconds=30.0, preprocess=True):
        self.processor = processor
        self.max_samples = int(max_seconds * 16000)
        self.indices = []  # list of (shard_idx, row_group, offset)
        self.shard_paths = sorted(shard_paths)

        for si, path in enumerate(self.shard_paths):
            pf = pq.ParquetFile(path)
            total_rows = pf.metadata.num_rows
            n_rg = pf.metadata.num_row_groups
            rows_per_rg = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
            off = 0
            for rg_idx in range(n_rg):
                for row_in_rg in range(rows_per_rg[rg_idx]):
                    self.indices.append((si, rg_idx, row_in_rg))
            pf = None  # release

        self.total_rows = len(self.indices)
        self._preprocessed_offsets = set()
        self._preprocessed = {}  # cache for (si, rg) -> full decoded batch
        self.preprocess = preprocess
        print(f"  {len(self.shard_paths)} shards, {self.total_rows:,} rows")

    def _load_row_group(self, si, rg):
        """Load and decode a full row group, cache it."""
        key = (si, rg)
        if key in self._preprocessed:
            return self._preprocessed[key]

        pf = pq.ParquetFile(self.shard_paths[si])
        batch = pf.read_row_group(rg, columns=["audio", "transcript"]).to_pydict()

        audios = []
        texts = []
        for i in range(len(batch["transcript"])):
            wav_bytes = batch["audio"][i]["bytes"]
            data, sr = sf.read(io.BytesIO(wav_bytes))
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            audios.append(data.astype(np.float32))
            texts.append(batch["transcript"][i])

        self._preprocessed[key] = (audios, texts)
        pf = None
        return audios, texts

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        si, rg, offset = self.indices[idx]
        audios, texts = self._load_row_group(si, rg)

        audio_arr = audios[offset]
        text = texts[offset]

        # Filter: skip if too long (return None) — caller handles this
        if audio_arr.shape[0] > self.max_samples:
            return {"input_features": None, "labels": None, "_skip": True}

        input_features = self.processor.feature_extractor(
            audio_arr, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels, "_skip": False}

    def clear_cache(self):
        """Release decoded row groups from RAM."""
        self._preprocessed.clear()
        gc.collect()


# ── Collator ────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter out skipped items
        features = [f for f in features if not f.get("_skip", False)]
        if not features:
            return {}

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pred, processor, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", choices=["tiny", "small", "medium", "large-v3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-audio-sec", type=float, default=30.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_id = f"openai/whisper-{args.model}"
    print(f"═══ Model: {model_id} ═══")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {props.total_memory / 1e9:.1f} GB")

    # ── Processor ──────────────────────────────────────────────────────
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens("en")

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.suppress_tokens = []

    if args.freeze_encoder:
        model.freeze_encoder()
        print("  Encoder frozen")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total, {n_train:,} trainable")

    # ── Datasets ───────────────────────────────────────────────────────
    print("\n═══ Loading datasets ═══")

    def find_shards(prefix):
        return sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith(prefix))

    train_ds = SPGISpeechDataset(find_shards("train"), processor,
                                  max_seconds=args.max_audio_sec)
    val_full = SPGISpeechDataset(find_shards("validation"), processor,
                                 max_seconds=args.max_audio_sec)
    test_full = SPGISpeechDataset(find_shards("test"), processor,
                                  max_seconds=args.max_audio_sec)

    # Subset eval/test for speed
    def subset_indices(ds, n):
        if len(ds) <= n:
            return list(range(len(ds)))
        rng = np.random.default_rng(42)
        return sorted(rng.choice(len(ds), n, replace=False).tolist())

    val_idx = subset_indices(val_full, args.eval_samples)
    test_idx = subset_indices(test_full, args.test_samples)
    val_ds = Subset(val_full, val_idx)
    test_ds = Subset(test_full, test_idx)

    print(f"  Train: {len(train_ds):,}")
    print(f"  Eval:  {len(val_idx)} (subset of {len(val_full):,})")
    print(f"  Test:  {len(test_idx)} (subset of {len(test_full):,})")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = load_metric("wer", trust_remote_code=True)

    # ── Training args ──────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # 0 because we use shared state in Dataset
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
    )

    # ── Train ──────────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    print(f"\n═══ Starting training ═══")
    print(f"  Batch: {args.batch_size}, grad_accum: {args.grad_accum}, effective: {effective_batch}")
    print(f"  Steps/epoch: ~{steps_per_epoch}")
    print(f"  LR: {args.lr}, epochs: {args.epochs}")
    print(f"  Eval: every {args.eval_steps} steps ({len(val_idx)} samples)")
    print(f"  Save: every {args.save_steps} steps")
    print(f"  FP16: True, gradient checkpoint: True")

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.time() - t0
    print(f"\n═══ Training done in {elapsed/60:.1f} min ═══")

    # Save final
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"  Saved to {final_dir}")

    # ── Test eval ──────────────────────────────────────────────────────
    print("\n═══ Test evaluation ═══")
    # Reload best checkpoint
    best_dir = os.path.join(OUTPUT_DIR, "checkpoint-*")
    # trainer automatically loads best via load_best_model_at_end
    predictions = trainer.predict(test_ds)
    test_metrics = compute_metrics(predictions, processor, metric)
    print(f"  Test WER ({len(test_idx)} samples): {test_metrics['wer']:.4f}")

    log_entry = {
        "model": model_id,
        "train_samples": len(train_ds),
        "test_wer": test_metrics["wer"],
        "elapsed_min": elapsed / 60,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "freeze_encoder": args.freeze_encoder,
        "max_audio_sec": args.max_audio_sec,
        "steps_per_epoch": steps_per_epoch,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"  Logged to {LOG_FILE}")
    print(f"\nDone. Eval in tensorboard: tensorboard --logdir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()