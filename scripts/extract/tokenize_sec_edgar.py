#!/usr/bin/env python3
"""Tokenize SEC-EDGAR filing parquet files into nanoGPT binary format.

Reads all .parquet files from /mnt/data/zz/datasets/sec-edgar/ (recursively,
across all filing-type subdirs: 10-K, 10-Q, 8-K, etc.).
Tokenizes 'text' column with GPT-2 BPE (tiktoken), produces:
  /mnt/data/zz/datasets/sec-edgar-tok/train_XXXXXX.npy  (100M tokens each)
  /mnt/data/zz/datasets/sec-edgar-tok/val_000000.npy    (first shard = val)

Memory-efficient: streams parquet in row-group batches, accumulates directly
into numpy uint16 array. Single-process to avoid memory duplication.

Usage:
    python3.11 scripts/extract/tokenize_sec_edgar.py

    python3.11 scripts/extract/tokenize_sec_edgar.py \
        --input-dir /mnt/data/zz/datasets/sec-edgar \
        --output-dir /mnt/data/zz/datasets/sec-edgar-tok
"""

import argparse
import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

SHARD_SIZE = int(1e8)  # 100M tokens per shard
TEXT_COLUMNS = ("text", "content", "article", "text_content", "body")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="/mnt/data/zz/datasets/sec-edgar")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/sec-edgar-tok")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Recursively find all parquet files across filing-type subdirs
    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        print(f"ERROR: No .parquet files in {args.input_dir}")
        print("Run download_sec_edgar.py first.")
        return

    total_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f"Found {len(parquet_files)} parquet files ({total_size / 1024**3:.1f} GB)")

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token  # 50256

    # Pre-allocate shard buffer in uint16 — 100M tokens = 200 MB
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
        fname = os.path.relpath(pq_path, args.input_dir)
        try:
            pf = pq.ParquetFile(pq_path)
        except Exception as e:
            print(f"  [skip] {fname}: {e}", flush=True)
            continue

        # Pick the first available text column
        names = pf.schema_arrow.names
        text_col = next((c for c in TEXT_COLUMNS if c in names), None)
        if text_col is None:
            print(f"  [skip] {fname}: no text column in {names}", flush=True)
            continue

        file_tokens = 0

        for batch in pf.iter_batches(columns=[text_col], batch_size=8192):
            for val in batch.column(text_col):
                if val is None or val.as_py() is None:
                    continue
                text = val.as_py()
                if not text or len(text.strip()) < 10:
                    continue
                tokens = enc.encode_ordinary(text)
                # EOT separator + tokens — write directly to numpy buffer
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
