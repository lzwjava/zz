#!/usr/bin/env python3
"""Tokenize codeparrot-clean parquet files (text col) -> nanoGPT binary format.

Reads all .parquet files from /mnt/data/zz/datasets/codeparrot-clean-nanochat/
Tokenizes 'text' column with GPT-2 BPE (tiktoken), produces:
  /mnt/data/zz/datasets/codeparrot-clean-tok/train_XXXXXX.npy  (100M tokens each)
  /mnt/data/zz/datasets/codeparrot-clean-tok/val_000000.npy

Memory-efficient: streams parquet in row-group batches, accumulates directly
into numpy uint16 array (no Python list phase).

Usage:
    python3.11 tokenize.py
"""

import argparse
import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

SHARD_SIZE = int(1e8)  # 100M tokens per shard


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="/mnt/data/zz/datasets/codeparrot-clean-nanochat")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/codeparrot-clean-tok")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquet_files:
        print(f"ERROR: No .parquet files in {args.input_dir}")
        print("Run convert.py first.")
        return

    total_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f"Found {len(parquet_files)} parquet files ({total_size / 1024**3:.1f} GB)")

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # 50256

    shard_buf = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    total_tokens = 0
    t0 = time.time()

    def flush_shard():
        nonlocal shard_index, token_count, total_tokens
        if token_count == 0:
            return
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(args.output_dir,
                                f"{split}_{shard_index:06d}.npy")
        np.save(filename, shard_buf[:token_count])
        print(f"  Wrote {filename} ({token_count:,} tokens)", flush=True)
        total_tokens += token_count
        shard_index += 1
        token_count = 0

    for fi, pq_path in enumerate(parquet_files):
        fname = os.path.basename(pq_path)
        pf = pq.ParquetFile(pq_path)
        file_tokens = 0

        for batch in pf.iter_batches(columns=["text"], batch_size=8192):
            for val in batch.column("text"):
                if val is None or val.as_py() is None:
                    continue
                text = val.as_py()
                if not text or len(text.strip()) < 10:
                    continue
                tokens = enc.encode_ordinary(text)
                shard_buf[token_count] = EOT
                token_count += 1
                file_tokens += 1
                if token_count >= SHARD_SIZE:
                    flush_shard()
                for t in tokens:
                    shard_buf[token_count] = t
                    token_count += 1
                    file_tokens += 1
                    if token_count >= SHARD_SIZE:
                        flush_shard()

        elapsed = time.time() - t0
        running_total = total_tokens + token_count
        rate = running_total / elapsed if elapsed > 0 else 0
        print(f"  [{fi+1}/{len(parquet_files)}] {fname}: "
              f"{file_tokens:,} tokens | "
              f"{running_total / 1e9:.2f}B total | "
              f"{rate / 1e6:.1f}M tok/s", flush=True)

    flush_shard()

    elapsed = time.time() - t0
    print(f"\nDone. {shard_index} shards, ~{total_tokens / 1e9:.2f}B tokens, "
          f"{elapsed / 60:.1f} min")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()